import polars as pl
import pytest

from gram_quant.engine.metrics import MetricsEngine, MetricsError


def test_cumulative_return_and_negative():
    df = pl.DataFrame(
        {
            "event_id": ["e1"] * 4,
            "relative_minute": [-1, 0, 1, 2],
            "close": [100.0, 105.0, 90.0, 100.0],
            "volume": [10.0, 10.0, 10.0, 10.0],
        }
    )

    engine = MetricsEngine()
    result = engine.calculate_metrics(df)

    returns = result["cumulative_return"].to_list()
    assert returns[0] == pytest.approx(0.0)
    assert returns[1] == pytest.approx(0.05)
    assert returns[2] == pytest.approx(-0.10)
    assert returns[3] == pytest.approx(0.0)


def test_multiple_events_leakage():
    df = pl.DataFrame(
        {
            "event_id": ["e1", "e1", "e2", "e2"],
            "relative_minute": [-1, 0, -1, 0],
            "close": [100.0, 110.0, 200.0, 210.0],
            "volume": [10.0, 10.0, 10.0, 10.0],
        }
    )

    engine = MetricsEngine()
    result = engine.calculate_metrics(df)

    e1_return = result.filter((pl.col("event_id") == "e1") & (pl.col("relative_minute") == 0))[
        "cumulative_return"
    ][0]

    e2_return = result.filter((pl.col("event_id") == "e2") & (pl.col("relative_minute") == 0))[
        "cumulative_return"
    ][0]

    assert e1_return == pytest.approx(0.10)
    assert e2_return == pytest.approx(0.05)


def test_missing_baseline():
    df = pl.DataFrame(
        {
            "event_id": ["e1"] * 3,
            "relative_minute": [-2, 0, 1],
            "close": [100.0, 105.0, 110.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )

    engine = MetricsEngine()
    with pytest.raises(MetricsError, match="Missing baseline"):
        engine.calculate_metrics(df)


def test_duplicate_baseline():
    df = pl.DataFrame(
        {
            "event_id": ["e1"] * 3,
            "relative_minute": [-1, -1, 0],
            "close": [100.0, 100.0, 105.0],
            "volume": [10.0, 10.0, 10.0],
        }
    )

    engine = MetricsEngine()
    with pytest.raises(MetricsError, match="Duplicate baseline"):
        engine.calculate_metrics(df)


def test_zero_volume_baseline():
    df = pl.DataFrame(
        {
            "event_id": ["e1"] * 3,
            "relative_minute": [-2, -1, 0],
            "close": [100.0, 100.0, 105.0],
            "volume": [0.0, 0.0, 50.0],
        }
    )

    engine = MetricsEngine()
    result = engine.calculate_metrics(df)

    t0_spike = result.filter(pl.col("relative_minute") == 0)["volume_spike"][0]
    assert t0_spike is None


# ==============================================================================
# Market-Adjusted AR (Clean AR) Tests
# ==============================================================================


def test_market_adjusted_car_basic():
    """
    Перевірка обчислення Clean AR та CAR через OLS-регресію проти ринку.

    Estimation Window:
      returns_asset  = [0.02, 0.04, 0.06]
      returns_market = [0.01, 0.02, 0.03]
      -> OLS OLS: beta = 2.0, alpha = 0.0

    Event Window:
      returns_asset  = [0.05, 0.08]
      returns_market = [0.02, 0.03]
      -> AR_0 = 0.05 - (0.0 + 2.0 * 0.02) = 0.01
      -> AR_1 = 0.08 - (0.0 + 2.0 * 0.03) = 0.02
      -> CAR_0 = 0.01
      -> CAR_1 = 0.01 + 0.02 = 0.03
    """
    estimation_df = pl.DataFrame(
        {
            "returns_asset": [0.02, 0.04, 0.06],
            "returns_market": [0.01, 0.02, 0.03],
        }
    )

    event_df = pl.DataFrame(
        {
            "event_id": ["e1", "e1"],
            "relative_minute": [-1, 0],
            "returns_asset": [0.05, 0.08],
            "returns_market": [0.02, 0.03],
        }
    )

    engine = MetricsEngine()
    result, alpha, beta = engine.calculate_market_adjusted_metrics(
        estimation_df=estimation_df,
        event_df=event_df,
        asset_col="returns_asset",
        market_col="returns_market",
    )

    assert beta == pytest.approx(2.0, abs=1e-5)
    assert alpha == pytest.approx(0.0, abs=1e-5)

    ar_values = result["abnormal_return"].to_list()
    car_values = result["car"].to_list()

    assert ar_values[0] == pytest.approx(0.01, abs=1e-5)
    assert ar_values[1] == pytest.approx(0.02, abs=1e-5)

    assert car_values[0] == pytest.approx(0.01, abs=1e-5)
    assert car_values[1] == pytest.approx(0.03, abs=1e-5)


def test_market_adjusted_missing_columns():
    """Перевіряє викидання помилки при відсутності необхідних колонок."""
    estimation_df = pl.DataFrame({"returns_asset": [0.01, 0.02]})
    event_df = pl.DataFrame({"returns_asset": [0.01, 0.02]})

    engine = MetricsEngine()
    with pytest.raises(MetricsError, match="Missing required columns"):
        engine.calculate_market_adjusted_metrics(
            estimation_df=estimation_df,
            event_df=event_df,
            asset_col="returns_asset",
            market_col="returns_market",
        )