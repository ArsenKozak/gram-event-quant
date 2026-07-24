import asyncio

from telethon import TelegramClient

from gram_quant.core.config import settings


async def main():
    if not settings.telegram_api_id or not settings.telegram_api_hash:
        print("❌ Telegram API credentials are missing in .env!")
        return

    api_hash = settings.telegram_api_hash.get_secret_value()
    session_path = "data/raw/telegram_session"
    print(f"🔑 Authorizing Telegram Client (API ID: {settings.telegram_api_id})...")

    async with TelegramClient(session_path, settings.telegram_api_id, api_hash) as client:
        me = await client.get_me()
        print(f"✅ Successfully authorized as: {me.first_name} (@{me.username})")


if __name__ == "__main__":
    asyncio.run(main())
