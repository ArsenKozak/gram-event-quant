import polars as pl
import pytest

from gram_quant.engine.metrics import MetricsEngine, MetricsError


def test_cumulative_return_and_negative():
    df = pl.DataFrame({
        "event_id": ["e1"] * 4,
        "relative_minute": [-1, 0, 1, 2],
        "close": [100.0, 105.0, 90.0, 100.0],
        "volume": [10.0, 10.0, 10.0, 10.0],
    })
    
    engine = MetricsEngine()
    result = engine.calculate_metrics(df)
    
    returns = result["cumulative_return"].to_list()
    assert returns[0] == pytest.approx(0.0)
    assert returns[1] == pytest.approx(0.05)
    assert returns[2] == pytest.approx(-0.10)
    assert returns[3] == pytest.approx(0.0)


def test_multiple_events_leakage():
    df = pl.DataFrame({
        "event_id": ["e1", "e1", "e2", "e2"],
        "relative_minute": [-1, 0, -1, 0],
        "close": [100.0, 110.0, 200.0, 210.0],
        "volume": [10.0, 10.0, 10.0, 10.0],
    })
    
    engine = MetricsEngine()
    result = engine.calculate_metrics(df)
    
    e1_return = result.filter(
        (pl.col("event_id") == "e1") & (pl.col("relative_minute") == 0)
    )["cumulative_return"][0]
    
    e2_return = result.filter(
        (pl.col("event_id") == "e2") & (pl.col("relative_minute") == 0)
    )["cumulative_return"][0]
    
    assert e1_return == pytest.approx(0.10)
    assert e2_return == pytest.approx(0.05)


def test_missing_baseline():
    df = pl.DataFrame({
        "event_id": ["e1"] * 3,
        "relative_minute": [-2, 0, 1],
        "close": [100.0, 105.0, 110.0],
        "volume": [10.0, 10.0, 10.0],
    })
    
    engine = MetricsEngine()
    with pytest.raises(MetricsError, match="Missing baseline"):
        engine.calculate_metrics(df)


def test_duplicate_baseline():
    df = pl.DataFrame({
        "event_id": ["e1"] * 3,
        "relative_minute": [-1, -1, 0],
        "close": [100.0, 100.0, 105.0],
        "volume": [10.0, 10.0, 10.0],
    })
    
    engine = MetricsEngine()
    with pytest.raises(MetricsError, match="Duplicate baseline"):
        engine.calculate_metrics(df)


def test_zero_volume_baseline():
    df = pl.DataFrame({
        "event_id": ["e1"] * 3,
        "relative_minute": [-2, -1, 0],
        "close": [100.0, 100.0, 105.0],
        "volume": [0.0, 0.0, 50.0],
    })
    
    engine = MetricsEngine()
    result = engine.calculate_metrics(df)
    
    t0_spike = result.filter(pl.col("relative_minute") == 0)["volume_spike"][0]
    assert t0_spike is None
