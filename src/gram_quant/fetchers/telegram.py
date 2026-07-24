from datetime import UTC
from pathlib import Path

import polars as pl
from loguru import logger
from telethon import TelegramClient
from telethon.tl.types import Message


class TelegramFetcher:
    """
    Асинхронний парсер новинних повідомлень із публічних Telegram-каналів.
    Використовує Telethon для витягування історії постів.
    """

    def __init__(
        self,
        api_id: int | None = None,
        api_hash: str | None = None,
        session_name: str = "data/raw/telegram_session",
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.session_path = str(Path(session_name))

    async def fetch_channel_messages(
        self,
        channel_username: str = "durov",
        limit: int = 100,
        min_id: int = 0,
    ) -> pl.DataFrame:
        """
        Витягує останні `limit` повідомлень з каналу та повертає їх у Polars DataFrame.
        """
        if not self.api_id or not self.api_hash:
            logger.warning(
                "Telegram API credentials (api_id/api_hash) not provided. "
                "Returning empty DataFrame."
            )
            return pl.DataFrame()

        async with TelegramClient(self.session_path, self.api_id, self.api_hash) as client:
            logger.info(f"Connecting to Telegram API to fetch @{channel_username}...")
            records = []
            
            async for message in client.iter_messages(channel_username, limit=limit, min_id=min_id):
                if isinstance(message, Message) and message.text:
                    # Час публікації завжди в UTC у Telethon
                    msg_dt = message.date.astimezone(UTC)
                    records.append({
                        "message_id": message.id,
                        "datetime": msg_dt,
                        "timestamp": int(msg_dt.timestamp() * 1000),
                        "author": channel_username,
                        "text": message.text,
                        "views": message.views or 0,
                        "forwards": message.forwards or 0,
                        "url": f"https://t.me/{channel_username}/{message.id}",
                    })

            if not records:
                logger.warning(f"No messages found in @{channel_username}")
                return pl.DataFrame()

            df = pl.DataFrame(records).sort("datetime")
            logger.success(f"Successfully fetched {len(df)} posts from @{channel_username}")
            return df
