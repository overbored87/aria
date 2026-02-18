"""Central configuration — reads env vars once, validates, exports."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # Telegram
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    allowed_user_id: int = int(os.getenv("ALLOWED_USER_ID", "0"))

    # Anthropic
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    claude_model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 1024
    token_budget: int = 8000       # max tokens for conversation history
    max_messages: int = 50         # max messages to fetch for context
    summary_threshold: int = 40    # trigger summary after N messages in 24h

    # Supabase
    supabase_url: str = os.getenv("SUPABASE_URL", "")
    supabase_key: str = os.getenv("SUPABASE_SERVICE_KEY", "")

    # User defaults
    user_timezone: str = os.getenv("USER_TIMEZONE", "Asia/Singapore")
    user_name: str = "Kieran"
    quiet_start: int = 23   # 11 PM
    quiet_end: int = 7      # 7 AM
    max_proactive_per_day: int = 5

    # Serper.dev (optional — web search disabled if not set)
    serper_api_key: str = os.getenv("SERPER_API_KEY", "")

    # Dashboard Supabase (optional — separate DB for personal dashboard)
    dashboard_supabase_url: str = os.getenv("DASHBOARD_SUPABASE_URL", "")
    dashboard_supabase_key: str = os.getenv("DASHBOARD_SUPABASE_KEY", "")

    # App
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    def validate(self) -> None:
        missing = []
        if not self.telegram_token:
            missing.append("TELEGRAM_BOT_TOKEN")
        if not self.allowed_user_id:
            missing.append("ALLOWED_USER_ID")
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        if not self.supabase_url:
            missing.append("SUPABASE_URL")
        if not self.supabase_key:
            missing.append("SUPABASE_SERVICE_KEY")
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")


cfg = Config()

