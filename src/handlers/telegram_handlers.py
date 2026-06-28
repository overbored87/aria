"""Telegram message handlers — routing, auth, response flow."""

from __future__ import annotations

import asyncio
import base64
import re

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
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
    get_pending_wiki_edits_from_response,
    remove_pending_wiki_edit,
)
from src.services import dashboard as dashboard_svc
from src.services.summarizer import maybe_summarize

# Module-level pending wiki edit (only one at a time)
_pending_wiki_edit: dict | None = None


def register_handlers(app: Application) -> None:
    """Attach all handlers to the Telegram Application."""
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("memories", _cmd_memories))
    app.add_handler(CommandHandler("clear", _cmd_clear))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(CommandHandler("approve", _cmd_approve))
    app.add_handler(CommandHandler("reject", _cmd_reject))
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

    # Generate response
    response_text = generate_response(user_id, text)

    # Save both messages AFTER generation (so history doesn't double-include the current message)
    save_message(user_id, "user", text, metadata={
        "message_id": update.message.message_id,
        "chat_id": update.effective_chat.id,
    })
    save_message(user_id, "assistant", response_text)

    # Extract images, then split into message chunks
    text_without_images, images = parse_images(response_text)
    await _send_split_response(update, text_without_images, images)

    # Send wiki edit previews with approval buttons
    wiki_edits = get_pending_wiki_edits_from_response()
    for edit in wiki_edits:
        await _send_wiki_preview(update.effective_chat, edit)

    # Background: check if we need to summarize
    await maybe_summarize(user_id)


# ─── Photo Handler ───────────────────────────────────────────


async def _handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process incoming photos — download, encode, send to Claude for vision."""
    if not _is_authorized(update):
        uid = update.effective_user.id if update.effective_user else "unknown"
        await update.message.reply_text(f"I'm spoken for 😏 (your ID: {uid})")
        return

    user = update.effective_user
    user_id = user.id
    caption = update.message.caption or ""

    log.info(f"Photo from {user.first_name}: caption='{caption[:60]}...'")

    ensure_user(user)

    # Get the highest resolution photo
    photo = update.message.photo[-1]  # last = largest
    photo_file = await photo.get_file()

    # Download to bytes
    photo_bytes = await photo_file.download_as_bytearray()
    b64_data = base64.b64encode(bytes(photo_bytes)).decode("utf-8")

    # Determine media type (Telegram photos are always JPEG)
    media_type = "image/jpeg"

    log.info(f"Photo downloaded: {len(photo_bytes)} bytes, sending to Claude")

    # Show typing
    await update.effective_chat.send_action("typing")

    # Generate response with vision
    image_data = {"base64": b64_data, "media_type": media_type}
    response_text = generate_response(user_id, caption, image_data=image_data)

    # Save both messages AFTER generation
    save_message(user_id, "user", f"[Sent a photo] {caption}".strip(), metadata={
        "message_id": update.message.message_id,
        "chat_id": update.effective_chat.id,
        "has_image": True,
    })
    save_message(user_id, "assistant", response_text)

    # Send split response
    text_without_images, images = parse_images(response_text)
    await _send_split_response(update, text_without_images, images)

    # Send wiki edit previews
    wiki_edits = get_pending_wiki_edits_from_response()
    for edit in wiki_edits:
        await _send_wiki_preview(update.effective_chat, edit)

    await maybe_summarize(user_id)


# ─── Wiki Edit Preview & Approval ────────────────────────────


async def _send_wiki_preview(chat, edit: dict) -> None:
    """Send a truncated wiki edit preview with /approve /reject instructions."""
    global _pending_wiki_edit
    _pending_wiki_edit = edit

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
                f"{action} wiki page: *{title}*\n"
                f"Slug: `{edit['slug']}`\n\n"
                f"This will permanently delete the page.\n\n"
                f"/approve to confirm • /reject to cancel"
            )
        else:
            # Truncate to ~300 chars for preview
            content_preview = edit["content"][:300]
            if len(edit["content"]) > 300:
                content_preview += "..."

            preview_text = (
                f"{action} wiki page: *{title}*\n"
                f"Slug: `{edit['slug']}`\n"
                f"Content: {len(edit['content'])} chars\n\n"
                f"Preview:\n{content_preview}\n\n"
                f"/approve to save • /reject to discard"
            )

        try:
            await chat.send_message(preview_text, parse_mode="Markdown")
        except Exception:
            await chat.send_message(preview_text.replace("*", "").replace("`", ""))

    except Exception as e:
        log.error(f"Failed to send wiki preview: {e}")


async def _cmd_approve(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Approve a pending wiki edit."""
    global _pending_wiki_edit

    if not _is_authorized(update):
        return

    if not _pending_wiki_edit:
        await update.message.reply_text("Nothing pending to approve.")
        return

    edit = _pending_wiki_edit
    user_id = cfg.allowed_user_id
    success = False

    if edit["type"] == "create":
        result = dashboard_svc.create_wiki_page(
            user_id=user_id,
            title=edit.get("title") or edit["slug"],
            slug=edit["slug"],
            content=edit["content"],
        )
        if result is None:
            # Upsert failed — try plain update
            result = dashboard_svc.update_wiki_page(
                slug=edit["slug"],
                content=edit["content"],
                title=edit.get("title"),
            )
        success = result is not None

    elif edit["type"] == "update":
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
        success = result is not None

    elif edit["type"] == "delete":
        result = dashboard_svc.delete_wiki_page(slug=edit["slug"])
        success = result

    if success:
        action = {"create": "created", "update": "updated", "delete": "deleted"}[edit["type"]]
        await update.message.reply_text(f"✅ Wiki page '{edit.get('title') or edit['slug']}' {action}!")
        log.info(f"Wiki {edit['type']} approved: {edit['slug']}")
    else:
        await update.message.reply_text("❌ Failed to save. Check logs.")
        log.error(f"Wiki {edit['type']} failed: {edit['slug']}")

    remove_pending_wiki_edit(edit["id"])
    _pending_wiki_edit = None


async def _cmd_reject(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Reject a pending wiki edit."""
    global _pending_wiki_edit

    if not _is_authorized(update):
        return

    if not _pending_wiki_edit:
        await update.message.reply_text("Nothing pending to reject.")
        return

    edit = _pending_wiki_edit
    remove_pending_wiki_edit(edit["id"])
    _pending_wiki_edit = None

    await update.message.reply_text("❌ Wiki edit discarded.")
    log.info(f"Wiki {edit['type']} rejected: {edit['slug']}")


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

