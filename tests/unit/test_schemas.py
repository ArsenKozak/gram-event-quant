from datetime import datetime
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from gram_quant.core.schemas import NewsEventRaw

KYIV_TZ = ZoneInfo("Europe/Kyiv")


def test_news_event_schema_validation_success():
    """Позитивний тест валідації схеми з приблизним порівнянням float."""
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
    assert event.price_before == pytest.approx(0.015)


def test_news_event_schema_invalid_importance():
    """Негативний тест: importance > 5 має викликати ValidationError."""
    kyiv_time = datetime(2024, 5, 10, 15, 30, tzinfo=KYIV_TZ)
    
    with pytest.raises(ValidationError):
        NewsEventRaw(
            event_time_kyiv=kyiv_time,
            text="Test",
            importance=10,  # За межами 1..5
            price_before=0.015,
            btc_before=62000.0,
        )


def test_news_event_schema_invalid_price():
    """Негативний тест: ціна <= 0 має викликати ValidationError."""
    kyiv_time = datetime(2024, 5, 10, 15, 30, tzinfo=KYIV_TZ)
    
    with pytest.raises(ValidationError):
        NewsEventRaw(
            event_time_kyiv=kyiv_time,
            text="Test",
            importance=3,
            price_before=-0.05,  # Негативна ціна
            btc_before=62000.0,
        )
