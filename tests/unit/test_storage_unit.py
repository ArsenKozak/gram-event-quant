from datetime import UTC, datetime

import polars as pl

from gram_quant.core.storage import ParquetStorage


def test_parquet_storage_save_and_load(tmp_path):
    """Тестування збереження, кешування та дедуплікації Parquet свічок."""
    storage = ParquetStorage(base_dir=tmp_path)

    # Синтетичний датасет свічок
    data_1 = pl.DataFrame(
        {
            "timestamp": [1700000000000, 1700000060000],
            "open": [2.1, 2.2],
            "high": [2.15, 2.25],
            "low": [2.08, 2.18],
            "close": [2.14, 2.22],
            "volume": [1000.0, 1500.0],
            "turnover": [2140.0, 3330.0],
            "datetime": [
                datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
                datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
            ],
        }
    )

    # 1. Зберігаємо
    saved_path = storage.save_candles(data_1, symbol="TONUSDT", interval="1")
    assert saved_path.exists()

    # 2. Зчитуємо з кешу
    loaded_df = storage.load_candles(symbol="TONUSDT", interval="1")
    assert len(loaded_df) == 2
    assert loaded_df["close"][0] == 2.14

    # 3. Додаємо нові дані з перекриттям (дублікатом)
    data_2 = pl.DataFrame(
        {
            "timestamp": [1700000060000, 1700000120000],  # Перша свічка — дублікат
            "open": [2.2, 2.22],
            "high": [2.25, 2.30],
            "low": [2.18, 2.21],
            "close": [2.22, 2.28],
            "volume": [1500.0, 2000.0],
            "turnover": [3330.0, 4560.0],
            "datetime": [
                datetime(2024, 1, 1, 12, 1, tzinfo=UTC),
                datetime(2024, 1, 1, 12, 2, tzinfo=UTC),
            ],
        }
    )

    storage.save_candles(data_2, symbol="TONUSDT", interval="1")
    merged_df = storage.load_candles(symbol="TONUSDT", interval="1")

    # Перевірка: дублікат видалено, загалом має бути 3 свічки
    assert len(merged_df) == 3
