"""Telegram message handlers — routing, auth, response flow."""

from __future__ import annotations

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
from src.services.claude_ai import generate_response
from src.services.summarizer import maybe_summarize


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

    # Save assistant response
    save_message(user_id, "assistant", response_text)

    # Send response — try with Markdown first, fallback to plain
    try:
        await update.message.reply_text(response_text, parse_mode="Markdown")
    except Exception:
        try:
            await update.message.reply_text(response_text)
        except Exception as e:
            log.error(f"Failed to send message: {e}")
            await update.message.reply_text("Something went wrong sending my response. Try again?")

    # Background: check if we need to summarize
    await maybe_summarize(user_id)
