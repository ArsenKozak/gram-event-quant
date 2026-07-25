from datetime import UTC, datetime, timedelta
from zoneinfo import ZoneInfo

import polars as pl
import pytest

from gram_quant.core.schemas import NewsEventRaw
from gram_quant.engine.event_slicer import EventSlicer, EventWindowError


@pytest.fixture
def sample_event() -> NewsEventRaw:
    return NewsEventRaw(
        event_id="test_event_001",
        event_time_kyiv=datetime(2024, 1, 1, 14, 0, 0, tzinfo=ZoneInfo("Europe/Kyiv")),
        author="Pavel Durov",
        source="telegram",
        text="TON to the moon!",
        importance=3,
        price_before=2.0,
        btc_before=42000.0,
    )


@pytest.fixture
def sample_ohlcv_data() -> pl.DataFrame:
    start_time = datetime(2024, 1, 1, 10, 0, 0, tzinfo=UTC)
    TOTAL_MINUTES = 301

    timestamps = []
    prices = []

    gap_start = datetime(2024, 1, 1, 12, 1, 0, tzinfo=UTC)
    gap_end = datetime(2024, 1, 1, 12, 5, 0, tzinfo=UTC)

    for i in range(TOTAL_MINUTES):
        dt = start_time + timedelta(minutes=i)
        # Штучна дірка: пропускаємо 5 хвилин після T0
        if gap_start <= dt <= gap_end:
            continue
        timestamps.append(dt)
        prices.append(2.0 + (i * 0.001))

    return pl.DataFrame({
        "timestamp": timestamps,
        "open": prices,
        "high": [p + 0.01 for p in prices],
        "low": [p - 0.01 for p in prices],
        "close": prices,
        "volume": [1000.0] * len(prices),
    })


def test_slice_event_window_full_gap_and_ohlc(sample_event, sample_ohlcv_data):
    """Головний тест: перевірка відновлення всієї дірки, всіх OHLC полів та сортування."""
    slicer = EventSlicer(pre_event_minutes=60, post_event_minutes=120)
    sliced_df = slicer.slice_window(sample_event, sample_ohlcv_data)

    assert len(sliced_df) == 181
    assert sliced_df["relative_minute"][0] == -60
    assert sliced_df["relative_minute"][-1] == 120
    assert sliced_df["timestamp"].is_sorted()

    filled_gap = sliced_df.filter(pl.col("relative_minute").is_in([1, 2, 3, 4, 5]))
    assert len(filled_gap) == 5

    t0_row = sliced_df.filter(pl.col("relative_minute") == 0)
    t0_close = t0_row["close"][0]
    t0_open = t0_row["open"][0]
    t0_high = t0_row["high"][0]
    t0_low = t0_row["low"][0]

    assert all(filled_gap["volume"] == 0.0)
    assert all(filled_gap["close"] == t0_close)
    assert all(filled_gap["open"] == t0_open)
    assert all(filled_gap["high"] == t0_high)
    assert all(filled_gap["low"] == t0_low)


def test_event_out_of_range(sample_event, sample_ohlcv_data):
    """Подія знаходиться поза межами OHLCV даних."""
    slicer = EventSlicer(pre_event_minutes=60, post_event_minutes=120)

    # Змінюємо час події на 2025 рік, коли даних немає
    sample_event.event_time_kyiv = datetime(
        2025, 1, 1, 14, 0, 0, tzinfo=ZoneInfo("Europe/Kyiv")
    )

    with pytest.raises(EventWindowError, match="No candles found"):
        slicer.slice_window(sample_event, sample_ohlcv_data)
