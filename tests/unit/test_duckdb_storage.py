import polars as pl
import pytest
from gram_quant.storage.duckdb_store import DuckDBStore


@pytest.fixture
def temp_duckdb_store(tmp_path):
    db_file = tmp_path / "test.duckdb"
    store = DuckDBStore(db_path=str(db_file))
    yield store
    store.close()


def test_duckdb_parquet_view_and_query(temp_duckdb_store, tmp_path):
    parquet_path = tmp_path / "test_data.parquet"
    df_original = pl.DataFrame(
        {
            "timestamp": [1700000000, 1700000060],
            "symbol": ["TONUSDT", "TONUSDT"],
            "close": [2.50, 2.55],
        }
    )
    df_original.write_parquet(parquet_path)

    temp_duckdb_store.register_parquet_view("candles_test", str(parquet_path))

    query = "SELECT symbol, AVG(close) as avg_price FROM candles_test GROUP BY symbol"
    df_result = temp_duckdb_store.query_df(query)

    assert not df_result.is_empty()
    assert df_result["symbol"][0] == "TONUSDT"
    assert pytest.approx(df_result["avg_price"][0], 0.001) == 2.525


def test_duckdb_context_manager(tmp_path):
    db_file = tmp_path / "context_test.duckdb"
    with DuckDBStore(db_path=str(db_file)) as store:
        df = store.query_df("SELECT 1 as num")
        assert df["num"][0] == 1


def test_duckdb_invalid_view_name_guard(temp_duckdb_store, tmp_path):
    parquet_path = tmp_path / "test_data.parquet"
    pl.DataFrame({"a": [1]}).write_parquet(parquet_path)

    invalid_name = "view_name; DROP TABLE test;"
    with pytest.raises(ValueError, match="Invalid SQL view name"):
        temp_duckdb_store.register_parquet_view(invalid_name, str(parquet_path))
