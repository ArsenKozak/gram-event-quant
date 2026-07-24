from datetime import datetime
from zoneinfo import ZoneInfo

from gram_quant.core.schemas import NewsEventRaw

KYIV_TZ = ZoneInfo("Europe/Kyiv")
UTC_TZ = ZoneInfo("UTC")


def test_news_event_schema_validation():
    """Перевірка правильної валідації Pydantic схеми новинного евенту."""
    kyiv_time = datetime(2024, 5, 10, 15, 30, tzinfo=KYIV_TZ)
    
    event = NewsEventRaw(
        event_time_kyiv=kyiv_time,
        author="Pavel Durov",
        url="https://t.me/durov/250",
        text="GRAM is the future of TON ecosystem",
        importance=5,
        price_before=0.015,
        btc_before=62000.0,
    )

    assert event.author == "Pavel Durov"
    assert event.importance == 5
    assert event.price_before == 0.015


def test_timezone_conversion_kyiv_to_utc():
    """Перевірка конвертації часу з Київського у UTC."""
    kyiv_time = datetime(2024, 5, 10, 15, 30, tzinfo=KYIV_TZ)
    utc_time = kyiv_time.astimezone(UTC_TZ)

    # Літній час у травні: UTC = Kyiv - 3 hours
    assert utc_time.hour == 12
    assert utc_time.minute == 30
