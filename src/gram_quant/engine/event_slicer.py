from datetime import UTC, timedelta

import polars as pl

from gram_quant.core.schemas import NewsEventRaw


class EventWindowError(ValueError):
    """Викликається, коли неможливо побудувати вікно (наприклад, немає даних)."""
    pass


class EventSlicer:
    """Slices OHLCV time-series data around a specific event timestamp T0
    and ensures a continuous 1-minute grid using forward-fill.
    """

    def __init__(self, pre_event_minutes: int = 60, post_event_minutes: int = 120) -> None:
        self.pre_event_minutes = pre_event_minutes
        self.post_event_minutes = post_event_minutes

    def slice_window(self, event: NewsEventRaw, ohlcv: pl.DataFrame) -> pl.DataFrame:
        """Filters OHLCV candles, fills missing gaps, and appends relative time."""
        if ohlcv.is_empty():
            raise EventWindowError("No candles found: OHLCV dataframe is empty.")

        # Безпечна конвертація timezone-aware київського часу в UTC
        t0 = event.event_time_kyiv.astimezone(UTC)
        start_bound = t0 - timedelta(minutes=self.pre_event_minutes)
        end_bound = t0 + timedelta(minutes=self.post_event_minutes)

        # 1. Фільтрація часового вікна
        sliced_df = ohlcv.filter(
            (pl.col("timestamp") >= start_bound) & (pl.col("timestamp") <= end_bound)
        ).sort("timestamp")

        if sliced_df.is_empty():
            raise EventWindowError(
                f"No candles found in window [{start_bound} - {end_bound}]"
            )

        # 2. Вирівнювання сітки (Upsampling) кожну 1 хвилину
        sliced_df = sliced_df.upsample(time_column="timestamp", every="1m").with_columns(
            pl.col("open").fill_null(strategy="forward"),
            pl.col("high").fill_null(strategy="forward"),
            pl.col("low").fill_null(strategy="forward"),
            pl.col("close").fill_null(strategy="forward"),
            pl.col("volume").fill_null(0.0),
        )

        # 3. Розрахунок відносної хвилини
        sliced_df = sliced_df.with_columns(
            (
                (pl.col("timestamp") - t0).dt.total_seconds() // 60
            ).cast(pl.Int64).alias("relative_minute")
        )

        return sliced_df
