from pathlib import Path

import polars as pl
from loguru import logger


class ParquetStorage:
    """
    Менеджер Parquet-кешування ринкових свічок та новинних подій.
    Забезпечує блискавичне зчитування з диска замість повторних запитів до API.
    """

    def __init__(self, base_dir: str | Path = "data/raw"):
        self.base_dir = Path(base_dir)
        self.candles_dir = self.base_dir / "candles"
        self.candles_dir.mkdir(parents=True, exist_ok=True)

    def _get_candle_path(self, symbol: str, interval: str) -> Path:
        """Формує шлях до parquet-файлу для конкретного тикера."""
        return self.candles_dir / f"{symbol.upper()}_{interval}.parquet"

    def save_candles(self, df: pl.DataFrame, symbol: str, interval: str = "1") -> Path:
        """Зберігає або дозаписує (merge & deduplicate) свічки у Parquet."""
        if df.is_empty():
            logger.warning("Empty DataFrame provided to save_candles")
            return self._get_candle_path(symbol, interval)

        file_path = self._get_candle_path(symbol, interval)

        if file_path.exists():
            existing_df = pl.read_parquet(file_path)
            # Об'єднуємо та видаляємо дублікати за часом (datetime)
            combined_df = (
                pl.concat([existing_df, df])
                .unique(subset=["timestamp"])
                .sort("timestamp")
            )
        else:
            combined_df = df.sort("timestamp")

        combined_df.write_parquet(file_path, compression="snappy")
        logger.info(f"Saved {len(combined_df)} candles for {symbol} to {file_path}")
        return file_path

    def load_candles(
        self, symbol: str, interval: str = "1"
    ) -> pl.DataFrame:
        """Завантажує збережені свічки з Parquet-файлу."""
        file_path = self._get_candle_path(symbol, interval)
        if not file_path.exists():
            logger.warning(f"No cache file found at {file_path}")
            return pl.DataFrame()

        df = pl.read_parquet(file_path)
        logger.success(f"Loaded {len(df)} candles for {symbol} from Parquet cache")
        return df
