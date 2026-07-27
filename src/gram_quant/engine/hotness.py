import polars as pl


def calculate_hotness_score(
    df: pl.DataFrame,
    car_col: str = "car",
    vol_col: str = "volume_spike",
    weight_car: float = 0.5,
    weight_vol: float = 0.5,
) -> pl.DataFrame:
    """
    Обчислює Hotness Index для подій на основі абсолютного значення CAR та Volume Spike.

    Formula:
      Hotness = w_car * Scaled(|CAR|) + w_vol * Scaled(VolumeSpike)
    """
    # 1. Перевірка наявності необхідних колонок
    missing = [col for col in [car_col, vol_col] if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in DataFrame")

    if df.is_empty():
        return df.with_columns(pl.lit(0.0).alias("hotness_score"))

    # 2. Очищення null та NaN значень
    df_clean = df.with_columns(
        pl.col(car_col).fill_nan(0.0).fill_null(0.0).alias("_clean_car"),
        pl.col(vol_col).fill_nan(1.0).fill_null(1.0).alias("_clean_vol"),
    )

    # 3. Обчислення абсолютного значення CAR (бо дамп — це теж гаряча новина)
    df_clean = df_clean.with_columns(pl.col("_clean_car").abs().alias("_abs_car"))

    # 4. Нормалізація (MinMax scaling з зародишем безпеки від ділення на 0)
    max_car = df_clean["_abs_car"].max() or 0.0
    min_car = df_clean["_abs_car"].min() or 0.0
    range_car = max_car - min_car

    max_vol = df_clean["_clean_vol"].max() or 0.0
    min_vol = df_clean["_clean_vol"].min() or 0.0
    range_vol = max_vol - min_vol

    # Якщо розмах = 0 (наприклад, 1 елемент або всі однакові), ставимо 1.0 скор
    norm_car_expr = (pl.col("_abs_car") - min_car) / range_car if range_car > 0 else pl.lit(1.0)

    norm_vol_expr = (pl.col("_clean_vol") - min_vol) / range_vol if range_vol > 0 else pl.lit(1.0)

    result_df = df_clean.with_columns(
        (weight_car * norm_car_expr + weight_vol * norm_vol_expr).alias("hotness_score")
    )

    return result_df.drop(["_clean_car", "_clean_vol", "_abs_car"])
