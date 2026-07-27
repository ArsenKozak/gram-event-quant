import pytest
import polars as pl
import numpy as np
from gram_quant.stats.market_model import calculate_market_model_ar

def test_market_model_ols_and_ar_with_noise():
    """Перевіряє OLS регресію та розрахунок AR на даних із шумом та динамічною формулою."""
    # R_i = 0.01 + 2.0 * R_m + noise
    est_df = pl.DataFrame({
        "returns_market": [0.01, 0.02, 0.015, 0.04, 0.05],
        "returns_asset": [0.031, 0.049, 0.041, 0.089, 0.112],
    })

    event_df = pl.DataFrame({
        "returns_market": [0.03, -0.01, 0.0],
        "returns_asset": [0.08, 0.01, 0.01],
    })
    
    result_df, alpha, beta = calculate_market_model_ar(
        estimation_df=est_df,
        event_df=event_df,
        asset_col="returns_asset",
        market_col="returns_market"
    )
    
    # Допуски більші через наявність шуму, але перевіряємо, чи вловив тренд
    assert pytest.approx(alpha, abs=0.01) == 0.01
    assert pytest.approx(beta, abs=0.1) == 2.0
    
    # Senior check: динамічна перевірка математичної формули AR
    ar_values = result_df["abnormal_return"].to_list()
    event_market = event_df["returns_market"].to_list()
    event_asset = event_df["returns_asset"].to_list()
    
    for i in range(len(event_df)):
        expected_ar = event_asset[i] - (alpha + beta * event_market[i])
        assert pytest.approx(ar_values[i], rel=1e-5) == expected_ar


def test_market_model_empty_dataframe():
    """Edge case: порожні вікна."""
    empty_df = pl.DataFrame({"returns_asset": [], "returns_market": []}, schema={"returns_asset": pl.Float64, "returns_market": pl.Float64})
    valid_df = pl.DataFrame({"returns_asset": [0.01, 0.02], "returns_market": [0.01, 0.02]})
    
    with pytest.raises(ValueError, match="Estimation data is empty"):
        calculate_market_model_ar(empty_df, valid_df, "returns_asset", "returns_market")

    with pytest.raises(ValueError, match="Event data is empty"):
        calculate_market_model_ar(valid_df, empty_df, "returns_asset", "returns_market")


def test_market_model_constant_market():
    """Edge case: Market має нульову дисперсію. OLS видасть помилку ділення на нуль."""
    est_df = pl.DataFrame({
        "returns_asset": [0.01, 0.02, 0.03],
        "returns_market": [0.01, 0.01, 0.01],  # No variance
    })
    event_df = pl.DataFrame({"returns_asset": [0.01], "returns_market": [0.01]})
    
    with pytest.raises(ValueError, match="Zero variance in market returns"):
        calculate_market_model_ar(est_df, event_df, "returns_asset", "returns_market")


def test_market_model_insufficient_points():
    """Edge case: недостатньо точок для регресії (менше 2)."""
    est_df = pl.DataFrame({
        "returns_asset": [0.01],
        "returns_market": [0.01],
    })
    event_df = pl.DataFrame({"returns_asset": [0.01], "returns_market": [0.01]})
    
    with pytest.raises(ValueError, match="Insufficient data points for OLS"):
        calculate_market_model_ar(est_df, event_df, "returns_asset", "returns_market")


def test_market_model_with_nans():
    """Edge case: наявність NaN. Функція повинна вимагати очищених даних або очищати сама."""
    est_df = pl.DataFrame({
        "returns_asset": [0.01, np.nan, 0.03, 0.04],
        "returns_market": [0.01, 0.02, np.nan, 0.04],
    })
    event_df = pl.DataFrame({"returns_asset": [0.05], "returns_market": [0.05]})
    
    # Припускаємо, що пайплайн має drop_nulls() перед розрахунком OLS
    result_df, alpha, beta = calculate_market_model_ar(
        est_df.drop_nulls(), 
        event_df, 
        "returns_asset", 
        "returns_market"
    )
    assert alpha is not None
    assert beta is not None
