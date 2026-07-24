from pathlib import Path

import pytest

from gram_quant.core.config import settings
from gram_quant.fetchers.telegram import TelegramFetcher


@pytest.mark.integration
@pytest.mark.asyncio
async def test_telegram_fetcher_live():
    """Справжній інтеграційний тест з жорсткою перевіркою якості даних."""
    session_file = Path("data/raw/telegram_session.session")

    if not settings.telegram_api_id or not settings.telegram_api_hash:
        pytest.skip("Telegram API credentials not set in .env")

    if not session_file.exists():
        msg = "Telegram session not authorized. Run `uv run python scripts/auth_telegram.py`"
        pytest.skip(msg)

    fetcher = TelegramFetcher(
        api_id=settings.telegram_api_id,
        api_hash=settings.telegram_api_hash,
        session_name="data/raw/telegram_session",
    )

    for channel in ["durov", "toncoin"]:
        df = await fetcher.fetch_channel_messages(channel_username=channel, limit=3)

        # Строгі перевірки (Production Criteria)
        assert df.height > 0, f"Expected non-empty rows for @{channel}"
        assert df["text"].str.len_chars().sum() > 0, "Messages must contain non-empty text"
        assert df["datetime"].null_count() == 0, "Timestamps must not contain NULLs"
        assert "msg_id" in df.columns
