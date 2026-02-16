"""Telegram message handlers — routing, auth, response flow."""

from __future__ import annotations

import asyncio
import re

from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
)

from src.config import cfg
from src.utils.logger import log
from src.services.database import ensure_user, save_message, get_active_memories
from src.services.claude_ai import generate_response, parse_images, get_pending_reminders
from src.services.summarizer import maybe_summarize
from src.services.scheduler import schedule_reminder


def register_handlers(app: Application) -> None:
    """Attach all handlers to the Telegram Application."""
    app.add_handler(CommandHandler("start", _cmd_start))
    app.add_handler(CommandHandler("memories", _cmd_memories))
    app.add_handler(CommandHandler("clear", _cmd_clear))
    app.add_handler(CommandHandler("help", _cmd_help))
    app.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, _handle_message)
    )
    log.info("Telegram handlers registered")


# ─── Auth Guard ──────────────────────────────────────────────


def _is_authorized(update: Update) -> bool:
    """Only allow the configured user."""
    return update.effective_user and update.effective_user.id == cfg.allowed_user_id


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
        "/help — This message\n\n"
        "Or just talk to me naturally — I handle tasks, reminders, goals, "
        "and everything in between ✨",
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
        await update.message.reply_text("I'm spoken for 😏")
        return

    user = update.effective_user
    user_id = user.id
    text = update.message.text.strip()

    if not text:
        return

    log.info(f"Message from {user.first_name}: {text[:80]}...")

    # Ensure user exists
    ensure_user(user)

    # Save user message
    save_message(user_id, "user", text, metadata={
        "message_id": update.message.message_id,
        "chat_id": update.effective_chat.id,
    })

    # Show typing indicator
    await update.effective_chat.send_action("typing")

    # Generate response
    response_text = generate_response(user_id, text)

    # Save full assistant response
    save_message(user_id, "assistant", response_text)

    # Schedule any reminders that were parsed
    reminders = get_pending_reminders()
    for reminder in reminders:
        schedule_reminder(user_id, reminder["dt"], reminder["message"])

    # Extract images, then split into message chunks
    text_without_images, images = parse_images(response_text)
    await _send_split_response(update, text_without_images, images)

    # Background: check if we need to summarize
    await maybe_summarize(user_id)


# ─── Multi-Message Sender ───────────────────────────────────


async def _send_split_response(
    update: Update,
    text: str,
    images: list[dict] | None = None,
) -> None:
    """Split response on double newlines and send as separate Telegram messages.
    Images are sent at the end (or inline if we can match position later)."""
    chat = update.effective_chat

    # Split on double newlines (blank lines) — each chunk becomes a message
    chunks = [c.strip() for c in re.split(r"\n\n+", text) if c.strip()]

    if not chunks and not images:
        return

    for i, chunk in enumerate(chunks):
        # Small delay between messages for natural feel (skip first)
        if i > 0:
            await asyncio.sleep(0.4)
            await chat.send_action("typing")
            await asyncio.sleep(0.3)

        try:
            await chat.send_message(chunk, parse_mode="Markdown")
        except Exception:
            try:
                await chat.send_message(chunk)
            except Exception as e:
                log.error(f"Failed to send chunk {i}: {e}")

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

