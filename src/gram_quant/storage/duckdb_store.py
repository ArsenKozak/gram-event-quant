from pathlib import Path
from typing import Self
import duckdb
import polars as pl
from loguru import logger


class DuckDBStore:
    """
    OLAP-сховище на базі DuckDB для швидких аналітичних SQL-запитів
    над Parquet-кешем та ін-пам'ять Polars DataFrame.
    """

    def __init__(self, db_path: str = "data/processed/gram_quant.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = duckdb.connect(str(self.db_path))
        logger.info(f"Connected to DuckDB storage at {self.db_path}")

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def register_parquet_view(self, view_name: str, parquet_glob_pattern: str) -> None:
        """Реєструє Parquet-файли як SQL View без їх завантаження в RAM."""
        if not view_name.isidentifier():
            raise ValueError(f"Invalid SQL view name: '{view_name}'")

        query = (
            f"CREATE OR REPLACE VIEW {view_name} AS "
            f"SELECT * FROM read_parquet('{parquet_glob_pattern}')"
        )
        self.conn.execute(query)
        logger.info(f"Registered DuckDB view '{view_name}' from {parquet_glob_pattern}")

    def query_df(self, query: str) -> pl.DataFrame:
        """Виконує SQL-запит та повертає результат як Polars DataFrame."""
        arrow_table = self.conn.execute(query).to_arrow_table()
        return pl.from_arrow(arrow_table)

    def close(self) -> None:
        """Закриває з'єднання з базою даних."""
        self.conn.close()
        logger.info("Closed DuckDB connection.")
