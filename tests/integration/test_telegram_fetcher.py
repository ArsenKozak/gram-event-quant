from pathlib import Path

import pytest

from gram_quant.core.config import settings
from gram_quant.fetchers.telegram import TelegramFetcher


@pytest.mark.asyncio
async def test_telegram_fetcher_live():
    """Справжній інтеграційний тест Telethon для довільних каналів."""
    session_file = Path("data/raw/telegram_session.session")
    
    if not settings.telegram_api_id or not settings.telegram_api_hash:
        pytest.skip("Telegram API credentials not set in .env")

    if not session_file.exists():
        pytest.skip("Telegram session not authorized yet. Run `uv run python scripts/auth_telegram.py` first.")

    fetcher = TelegramFetcher(
        api_id=settings.telegram_api_id,
        api_hash=settings.telegram_api_hash,
        session_name="data/raw/telegram_session"
    )

    # Тестуємо універсальність: витягуємо останні 2 пости з двох різних каналів
    for channel in ["durov", "toncoin"]:
        df = await fetcher.fetch_channel_messages(channel_username=channel, limit=2)
        assert not df.is_empty(), f"Failed to fetch posts from @{channel}"
        assert "text" in df.columns
        assert "datetime" in df.columns
        assert len(df) <= 2
