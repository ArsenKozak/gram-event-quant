from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl
from loguru import logger

from gram_quant.core.schemas import NewsEventRaw

KYIV_TZ = ZoneInfo("Europe/Kyiv")
UTC_TZ = ZoneInfo("UTC")


class ExcelIngestor:
    """Модуль завантаження та первинної обробки ручного Excel-датасету."""

    def __init__(self, file_path: str | Path):
        self.file_path = Path(file_path)

    def load_raw_dataframe(self) -> pl.DataFrame:
        """Зчитує Excel у Polars DataFrame."""
        if not self.file_path.exists():
            raise FileNotFoundError(f"Excel file not found at: {self.file_path}")

        logger.info(f"Loading Excel dataset from {self.file_path}")
        return pl.read_excel(self.file_path)

    def parse_and_clean(self, df: pl.DataFrame) -> list[NewsEventRaw]:
        """
        Очищає DataFrame, конвертує часові пояси (Kyiv -> UTC)
        та валідує кожен рядок через Pydantic.
        """
        events: list[NewsEventRaw] = []

        for row in df.iter_rows(named=True):
            try:
                # Очищення та приведення часу до UTC
                raw_time = row.get("time") or row.get("час") or row.get("event_time")

                # Guard Clause для уникнення AttributeError (тихі баги з None)
                if raw_time is None:
                    continue

                # Type Casting
                if isinstance(raw_time, str):
                    dt = datetime.fromisoformat(raw_time)
                elif isinstance(raw_time, datetime):
                    dt = raw_time
                else:
                    continue

                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=KYIV_TZ)

                # Зафіксуємо правильний Kyiv time для моделі
                kyiv_dt = dt.astimezone(KYIV_TZ)

                event = NewsEventRaw(
                    event_time_kyiv=kyiv_dt,
                    author=row.get("author", "Pavel Durov"),
                    url=row.get("url"),
                    text=str(row.get("text") or row.get("повідомлення") or ""),
                    importance=int(row.get("importance") or row.get("важкість") or 3),
                    price_before=float(row.get("price_before") or row.get("ціна_до") or 0.0),
                    btc_before=float(row.get("btc_before") or row.get("btc_до") or 1.0),
                    volume_before=row.get("volume"),
                    price_5m=row.get("price_5m"),
                    price_15m=row.get("price_15m"),
                    price_1h=row.get("price_1h"),
                    price_6h=row.get("price_6h"),
                    price_24h=row.get("price_24h"),
                    max_price=row.get("max_price"),
                    min_price=row.get("min_price"),
                    peak_time=row.get("peak_time"),
                    manual_return=row.get("return"),
                )
                events.append(event)
            except Exception as e:
                logger.warning(f"Skipping invalid row {row}: {e}")

        logger.success(f"Successfully validated {len(events)} events from Excel")
        return events
