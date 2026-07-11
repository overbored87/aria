"""Telegram message handlers — routing, auth, response flow."""

from __future__ import annotations

import asyncio
import base64

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from src.config import cfg, ALLOWED_USER_IDS
from src.utils.logger import log
from src.services.database import ensure_user, save_message, get_active_memories
from src.services.claude_ai import (
    generate_response,
    parse_images,
    remove_pending_wiki_edit,
)
from src.services import dashboard as dashboard_svc
from src.services.summarizer import maybe_summarize

# Wiki edits awaiting /approve or /reject (a single response can propose several)
_pending_approval: list[dict] = []


def register_handlers(app: Application) -> None:
    """Attach all handlers to the Telegram Application."""
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("memories", _cmd_memories))
    app.add_handler(CommandHandler("clear", _cmd_clear))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(CommandHandler("approve", _cmd_approve))
    app.add_handler(CommandHandler("reject", _cmd_reject))
    app.add_handler(CommandHandler("wiki_approve", _cmd_wiki_approve))
    app.add_handler(CommandHandler("wiki_reject", _cmd_wiki_reject))
    app.add_handler(CallbackQueryHandler(_cb_wiki, pattern=r"^wiki_(ok|no):"))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message)
    )
    app.add_handler(
        MessageHandler(filters.PHOTO, _handle_photo)
    )
    log.info("Telegram handlers registered")


# ─── Auth Guard ──────────────────────────────────────────────


def _is_authorized(update: Update) -> bool:
    """Only allow whitelisted users."""
    return update.effective_user and update.effective_user.id in ALLOWED_USER_IDS


# ─── Command Handlers ───────────────────────────────────────


async def _cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        await update.message.reply_text("Sorry, I'm a personal assistant — not taking new clients 😉")
        return

    ensure_user(update.effective_user)
    name = cfg.user_name

    await update.message.reply_text(
        f"Hey {name} 👋\n\n"
        f"I'm Aria, your personal assistant. I'm here to help you stay on top of "
        f"things, remember what matters, and maybe make your day a little better.\n\n"
        f"Just talk to me like you would a friend. I'll remember our conversations "
        f"and check in on you from time to time.\n\n"
        f"Type /help to see what I can do."
    )


async def _cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    await update.message.reply_text(
        "*Commands:*\n"
        "/start — Introduction\n"
        "/memories — What I remember about you\n"
        "/clear — Clear conversation (keeps memories)\n"
        "/approve — Approve a pending wiki edit\n"
        "/reject — Reject a pending wiki edit\n"
        "/help — This message\n\n"
        "Or just talk to me naturally — I can search the web, "
        "read and edit your wiki, and remember what matters ✨",
        parse_mode="Markdown",
    )


async def _cmd_memories(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    user_id = update.effective_user.id
    memories = get_active_memories(user_id, limit=20)

    if not memories:
        await update.message.reply_text(
            "I don't have any stored memories yet — we're just getting started! "
            "The more we talk, the more I'll remember about what matters to you."
        )
        return

    grouped: dict[str, list[str]] = {}
    for m in memories:
        cat = m["category"]
        grouped.setdefault(cat, []).append(m["content"])

    lines = ["*Here's what I remember about you:*\n"]
    for cat, items in grouped.items():
        emoji = {
            "personal": "👤", "preference": "⭐", "goal": "🎯",
            "task": "📋", "relationship": "💛", "habit": "🔄",
            "work": "💻", "health": "🏃", "interest": "🎮", "other": "📝",
        }.get(cat, "📝")
        lines.append(f"\n{emoji} *{cat.title()}*")
        for item in items:
            lines.append(f"  • {item}")

    await update.message.reply_text("\n".join(lines), parse_mode="Markdown")


async def _cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not _is_authorized(update):
        return

    await update.message.reply_text(
        "Conversation context refreshed. Your memories are still intact — "
        "I haven't forgotten anything important 😊"
    )


# ─── Main Message Handler ───────────────────────────────────


async def _handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming text messages."""
    if not _is_authorized(update):
        uid = update.effective_user.id if update.effective_user else "unknown"
        await update.message.reply_text(f"I'm spoken for 😏 (your ID: {uid})")
        return

    user = update.effective_user
    user_id = user.id
    text = update.message.text.strip()

    if not text:
        return

    log.info(f"Message from {user.first_name}: {text[:80]}...")

    # Ensure user exists
    ensure_user(user)

    # Show typing indicator
    await update.effective_chat.send_action("typing")

    # Generate response in a worker thread so the event loop stays responsive
    response_text, wiki_edits = await asyncio.to_thread(generate_response, user_id, text)

    # Save both messages AFTER generation (so history doesn't double-include the current message)
    save_message(user_id, "user", text, metadata={
        "message_id": update.message.message_id,
        "chat_id": update.effective_chat.id,
    })
    save_message(user_id, "assistant", response_text)

    # Extract images, then split into message chunks
    text_without_images, images = parse_images(response_text)
    await _send_split_response(update, text_without_images, images)

    # Send wiki edit previews with approval instructions
    await _queue_wiki_edits(update.effective_chat, wiki_edits)

    # Background: check if we need to summarize
    await maybe_summarize(user_id)


# ─── Photo Handler ───────────────────────────────────────────

# Telegram delivers an album (media group) as one update per photo, all sharing
# a media_group_id. Buffer them and flush shortly after the last one arrives so
# Claude sees the whole album in a single call instead of replying per photo.
_ALBUM_FLUSH_DELAY = 2.0  # seconds since last photo before processing
_albums: dict[str, dict] = {}


async def _download_photo(update: Update) -> dict:
    """Download the largest-resolution photo, resized to ≤1024px JPEG."""
    photo = update.message.photo[-1]  # last = largest
    photo_file = await photo.get_file()
    photo_bytes = await photo_file.download_as_bytearray()

    # Resize to max 1024px to reduce Claude vision token cost
    from PIL import Image
    import io
    img = Image.open(io.BytesIO(bytes(photo_bytes)))
    max_side = 1024
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        photo_bytes = buf.getvalue()
        log.info(f"Photo resized to {img.size}, {len(photo_bytes)} bytes")
    else:
        photo_bytes = bytes(photo_bytes)
        log.info(f"Photo unchanged: {img.size}, {len(photo_bytes)} bytes")

    b64_data = base64.b64encode(photo_bytes).decode("utf-8")
    # Telegram photos are always JPEG
    return {"base64": b64_data, "media_type": "image/jpeg"}


async def _handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming photos — single, or buffered into an album."""
    if not _is_authorized(update):
        uid = update.effective_user.id if update.effective_user else "unknown"
        await update.message.reply_text(f"I'm spoken for 😏 (your ID: {uid})")
        return

    user = update.effective_user
    caption = update.message.caption or ""

    log.info(f"Photo from {user.first_name}: caption='{caption[:60]}...'")

    ensure_user(user)

    group_id = update.message.media_group_id
    if group_id:
        # Buffer only (downloads happen at flush) so the debounce timer
        # measures actual arrival gaps, not download time.
        group = _albums.setdefault(group_id, {"updates": [], "captions": []})
        group["updates"].append(update)
        if caption:
            group["captions"].append(caption)
        # Debounce: restart the flush timer on each arrival
        if task := group.get("task"):
            task.cancel()
        group["task"] = asyncio.create_task(_flush_album(group_id))
        return

    image = await _download_photo(update)
    await _respond_to_photos(update, user.id, caption, [image])


async def _flush_album(group_id: str) -> None:
    """Wait out the debounce window, then process the buffered album."""
    try:
        await asyncio.sleep(_ALBUM_FLUSH_DELAY)
    except asyncio.CancelledError:
        return  # superseded by a newer photo in the same album
    group = _albums.pop(group_id, None)
    if not group:
        return
    updates = group["updates"]
    caption = " ".join(group["captions"])
    log.info(f"Album {group_id}: downloading {len(updates)} photos")
    images = [await _download_photo(u) for u in updates]
    update = updates[-1]
    await _respond_to_photos(update, update.effective_user.id, caption, images)


async def _respond_to_photos(
    update: Update, user_id: int, caption: str, images: list[dict]
) -> None:
    """Shared pipeline: generate with vision, save, send, queue wiki edits."""
    await update.effective_chat.send_action("typing")

    response_text, wiki_edits = await asyncio.to_thread(
        generate_response, user_id, caption, images=images
    )

    # Save both messages AFTER generation
    count = len(images)
    label = "[Sent a photo]" if count == 1 else f"[Sent {count} photos]"
    save_message(user_id, "user", f"{label} {caption}".strip(), metadata={
        "message_id": update.message.message_id,
        "chat_id": update.effective_chat.id,
        "has_image": True,
        "image_count": count,
    })
    save_message(user_id, "assistant", response_text)

    # Send split response
    text_without_images, response_images = parse_images(response_text)
    await _send_split_response(update, text_without_images, response_images)

    # Send wiki edit previews
    await _queue_wiki_edits(update.effective_chat, wiki_edits)

    await maybe_summarize(user_id)


# ─── Wiki Edit Preview & Approval ────────────────────────────


async def _queue_wiki_edits(chat, wiki_edits: list[dict]) -> None:
    """Queue wiki edits for approval and send a preview for each."""
    global _pending_approval
    if not wiki_edits:
        return
    _pending_approval = list(wiki_edits)
    for edit in wiki_edits:
        await _send_wiki_preview(chat, edit)
    if len(wiki_edits) > 1:
        await chat.send_message(
            f"{len(wiki_edits)} edits pending — /approve or /reject applies to all."
        )


async def _send_wiki_preview(chat, edit: dict) -> None:
    """Send a truncated wiki edit preview with /approve /reject instructions."""
    try:
        if edit["type"] == "create":
            action = "📝 Create"
        elif edit["type"] == "delete":
            action = "🗑️ Delete"
        else:
            action = "✏️ Update"

        title = edit.get("title") or edit["slug"]

        if edit["type"] == "delete":
            preview_text = (
                f"{action} *{title}* (`{edit['slug']}`)\n\n"
                f"/approve to confirm • /reject to cancel"
            )
        else:
            word_count = len(edit["content"].split())
            description = edit.get("description", "").strip()
            preview_text = (
                f"{action} *{title}* (`{edit['slug']}`), {word_count} words\n"
                + (f"_{description}_\n" if description else "")
                + f"\n/approve to save • /reject to discard"
            )

        try:
            await chat.send_message(preview_text, parse_mode="Markdown")
        except Exception:
            await chat.send_message(preview_text.replace("*", "").replace("`", ""))

    except Exception as e:
        log.error(f"Failed to send wiki preview: {e}")


def _apply_wiki_edit(edit: dict) -> bool:
    """Apply a single approved wiki edit. Returns success."""
    user_id = cfg.allowed_user_id

    if edit["type"] == "create":
        # create_wiki_page falls back to update internally if the slug exists
        result = dashboard_svc.create_wiki_page(
            user_id=user_id,
            title=edit.get("title") or edit["slug"],
            slug=edit["slug"],
            content=edit["content"],
        )
        return result is not None

    if edit["type"] == "update":
        result = dashboard_svc.update_wiki_page(
            slug=edit["slug"],
            content=edit["content"],
            title=edit.get("title"),
        )
        if result is None:
            # Page doesn't exist yet — create it instead
            result = dashboard_svc.create_wiki_page(
                user_id=user_id,
                title=edit.get("title") or edit["slug"],
                slug=edit["slug"],
                content=edit["content"],
            )
        return result is not None

    if edit["type"] == "delete":
        return dashboard_svc.delete_wiki_page(slug=edit["slug"])

    return False


async def _cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Approve all pending wiki edits."""
    global _pending_approval

    if not _is_authorized(update):
        return

    if not _pending_approval:
        await update.message.reply_text("Nothing pending to approve.")
        return

    edits, _pending_approval = _pending_approval, []
    action_word = {"create": "created", "update": "updated", "delete": "deleted"}
    lines = []

    for edit in edits:
        name = edit.get("title") or edit["slug"]
        success = await asyncio.to_thread(_apply_wiki_edit, edit)
        if success:
            lines.append(f"✅ '{name}' {action_word[edit['type']]}")
            log.info(f"Wiki {edit['type']} approved: {edit['slug']}")
        else:
            lines.append(f"❌ '{name}' failed — check logs")
            log.error(f"Wiki {edit['type']} failed: {edit['slug']}")
        remove_pending_wiki_edit(edit["id"])

    await update.message.reply_text("\n".join(lines))


async def _cmd_reject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reject all pending wiki edits."""
    global _pending_approval

    if not _is_authorized(update):
        return

    if not _pending_approval:
        await update.message.reply_text("Nothing pending to reject.")
        return

    edits, _pending_approval = _pending_approval, []
    for edit in edits:
        remove_pending_wiki_edit(edit["id"])
        log.info(f"Wiki {edit['type']} rejected: {edit['slug']}")

    count = len(edits)
    await update.message.reply_text(
        "❌ Wiki edit discarded." if count == 1 else f"❌ {count} wiki edits discarded."
    )


async def _cmd_wiki_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Approve a sidecar research draft (job-based — separate from /approve,
    which handles edits proposed live in an active chat).

    Prefer the inline Save button on Aria's notification; this typed command is
    a fallback. With no arg it acts on the latest pending draft, because tapping
    the command in Telegram drops the job id after the space."""
    if not _is_authorized(update):
        return
    from src.sidecar import jobs as jobstore

    job = jobstore.get_job(context.args[0]) if context.args else jobstore.latest_actionable_job()
    if not job or not job.get("edit"):
        await update.message.reply_text("No research draft waiting (it may have expired).")
        return
    edit = job["edit"]
    ok = await asyncio.to_thread(_apply_wiki_edit, edit)
    jobstore.drop_job(job["job_id"])
    await update.message.reply_text(
        f"{'✅ Saved' if ok else '❌ Failed'}: {edit.get('title') or edit['slug']}"
    )


async def _cmd_wiki_reject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Discard a sidecar research draft without saving it."""
    if not _is_authorized(update):
        return
    from src.sidecar import jobs as jobstore

    job = jobstore.get_job(context.args[0]) if context.args else jobstore.latest_actionable_job()
    if job:
        jobstore.drop_job(job["job_id"])
    await update.message.reply_text("Discarded. Nothing was saved.")


async def _cb_wiki(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the Save/Discard buttons on a research-draft notification.
    callback_data is 'wiki_ok:<job_id>' or 'wiki_no:<job_id>'."""
    query = update.callback_query
    if not _is_authorized(update):
        await query.answer("Not authorized", show_alert=True)
        return
    from src.sidecar import jobs as jobstore

    await query.answer()
    action, _, job_id = query.data.partition(":")

    if action == "wiki_no":
        jobstore.drop_job(job_id)
        await query.edit_message_text("❌ Discarded. Nothing was saved.")
        log.info(f"Research draft rejected via button: {job_id}")
        return

    job = jobstore.get_job(job_id)
    if not job or not job.get("edit"):
        await query.edit_message_text(
            "This draft expired (Aria restarted). Ask again and I'll re-run it."
        )
        return

    edit = job["edit"]
    ok = await asyncio.to_thread(_apply_wiki_edit, edit)
    jobstore.drop_job(job_id)
    name = edit.get("title") or edit["slug"]
    await query.edit_message_text(f"{'✅ Saved' if ok else '❌ Failed'}: {name}")
    log.info(f"Research draft {'saved' if ok else 'FAILED'} via button: {job_id}")


# ─── Multi-Message Sender ───────────────────────────────────


async def _send_split_response(
    update: Update,
    text: str,
    images: list[dict] | None = None,
) -> None:
    """Send response as a single message, followed by any images."""
    chat = update.effective_chat

    text = text.strip()
    if text:
        # Telegram max message length is 4096 chars
        chunks = [text[i:i+4096] for i in range(0, len(text), 4096)]
        for chunk in chunks:
            try:
                await chat.send_message(chunk, parse_mode="Markdown")
            except Exception:
                try:
                    await chat.send_message(chunk)
                except Exception as e:
                    log.error(f"Failed to send message: {e}")

    # Send any images
    for img in (images or []):
        try:
            await asyncio.sleep(0.3)
            await chat.send_photo(
                photo=img["url"],
                caption=img.get("caption"),
            )
        except Exception as e:
            log.error(f"Failed to send image {img['url']}: {e}")
            # If image fails, send the caption as text fallback
            if img.get("caption"):
                try:
                    await chat.send_message(f"📷 {img['caption']}")
                except Exception:
                    pass

