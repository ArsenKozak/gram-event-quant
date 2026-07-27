import asyncio
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
from loguru import logger
from pydantic import SecretStr
from telethon import TelegramClient
from telethon.errors import ChannelPrivateError, FloodWaitError, UsernameInvalidError


class TelegramFetcher:
    """
    Асинхронний фетчер для парсингу публічних Telegram-каналів.
    Підтримує обробку FloodWaitError та перевірку доступності каналу.
    """

    def __init__(
        self,
        api_id: int | None,
        api_hash: SecretStr | None,
        session_name: str = "data/raw/telegram_session",
    ):
        self.api_id = api_id
        self.api_hash = api_hash.get_secret_value() if api_hash else None
        self.session_path = str(Path(session_name))

    async def fetch_channel_messages(
        self,
        channel_username: str,
        limit: int | None = None,
        start_date: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Завантажує повідомлення з публічного Telegram-каналу.
        Якщо передано start_date, цикл припиняється при досягненні старіших повідомлень.
        """
        if not self.api_id or not self.api_hash:
            raise ValueError("Telegram API ID or API Hash is missing.")

        schema = {
            "msg_id": pl.Int64,
            "channel": pl.Utf8,
            "datetime": pl.Datetime("ms", "UTC"),
            "text": pl.Utf8,
            "views": pl.Int64,
            "forwards": pl.Int64,
        }

        async with TelegramClient(self.session_path, self.api_id, self.api_hash) as client:
            try:
                entity = await client.get_entity(channel_username)
            except (UsernameInvalidError, ValueError):
                logger.error(f"Channel username '{channel_username}' is invalid or not found.")
                return pl.DataFrame(schema=schema)
            except ChannelPrivateError:
                logger.error(f"Channel '{channel_username}' is private or restricted.")
                return pl.DataFrame(schema=schema)

            if start_date is not None and start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=UTC)

            records = []
            try:
                async for msg in client.iter_messages(entity, limit=limit):
                    if not msg.text:
                        continue

                    msg_date = msg.date
                    if msg_date is None:
                        continue

                    if start_date is not None:
                        if msg_date.tzinfo is None:
                            msg_date = msg_date.replace(tzinfo=UTC)
                        if msg_date < start_date:
                            logger.info(
                                "Reached Telegram start_date %s while fetching %s",
                                start_date,
                                channel_username,
                            )
                            break

                    records.append(
                        {
                            "msg_id": msg.id,
                            "channel": channel_username,
                            "datetime": msg_date,
                            "text": msg.text,
                            "views": msg.views or 0,
                            "forwards": msg.forwards or 0,
                        }
                    )
            except FloodWaitError as e:
                logger.warning(f"Telegram Rate Limit reached. Must wait {e.seconds} seconds.")
                await asyncio.sleep(e.seconds)

            if not records:
                return pl.DataFrame(schema=schema)

            return pl.DataFrame(records, schema=schema)
