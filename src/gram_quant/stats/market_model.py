from typing import Tuple
import polars as pl
import numpy as np

def calculate_market_model_ar(
    estimation_df: pl.DataFrame,
    event_df: pl.DataFrame,
    asset_col: str = "returns_asset",
    market_col: str = "returns_market",
) -> Tuple[pl.DataFrame, float, float]:
    """
    Оцінює Market Model через OLS регресію та розраховує аномальну дохідність (AR).

    Formula:
        R_it = alpha + beta * R_mt + epsilon_it
        AR_it = R_it - (alpha + beta * R_mt)
    """
    # 1. Валідація на порожні DataFrame
    if estimation_df.is_empty():
        raise ValueError("Estimation data is empty")
    if event_df.is_empty():
        raise ValueError("Event data is empty")

    # 2. Очищення від null ТА NaN (Polars розрізняє null i nan)
    clean_est = estimation_df.filter(
        pl.col(asset_col).is_not_null() 
        & pl.col(asset_col).is_not_nan() 
        & pl.col(market_col).is_not_null() 
        & pl.col(market_col).is_not_nan()
    )

    # 3. Перевірка кількості спостережень
    if clean_est.height < 2:
        raise ValueError("Insufficient data points for OLS")

    x = clean_est[market_col].to_numpy()
    y = clean_est[asset_col].to_numpy()

    # 4. Перевірка на нульову дисперсію ринку (константний ринок)
    var_x = np.var(x, ddof=1)
    if var_x == 0 or np.isnan(var_x):
        raise ValueError("Zero variance in market returns")

    # 5. Розрахунок OLS (Alpha та Beta)
    cov_xy = np.cov(x, y, ddof=1)[0, 1]
    beta = float(cov_xy / var_x)
    alpha = float(np.mean(y) - beta * np.mean(x))

    # 6. Векторизований розрахунок AR у event_df через Polars
    result_df = event_df.with_columns(
        (
            pl.col(asset_col) - (alpha + beta * pl.col(market_col))
        ).alias("abnormal_return")
    )

    return result_df, alpha, beta
