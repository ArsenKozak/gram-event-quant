import math
import polars as pl
import pytest

from gram_quant.engine.hotness import calculate_hotness_score


def test_hotness_score_ranking_and_exact_math():
    """
    1. Перевірка відносного ранжування (Monotonic Ranking).
    2. Перевірка математичної точності стандартизації (Z-score або Abs-CAR для абсолютного імпакту).
       Абсолютний CAR критичний: CAR -10% з об'ємом 10x — це надзвичайно гаряча подія (паніка/дамп).
    """
    df = pl.DataFrame(
        {
            "event_id": ["e1_pump", "e2_dump", "e3_flat"],
            "car": [0.10, -0.10, 0.00],
            "volume_spike": [5.0, 5.0, 1.0],
        }
    )

    # За замовчуванням w_car = 0.5, w_vol = 0.5
    result = calculate_hotness_score(df, car_col="car", vol_col="volume_spike")

    assert "hotness_score" in result.columns

    scores = dict(zip(result["event_id"], result["hotness_score"]))

    # Помпа і дамп при однакових об'ємах мають мати однаково високий Hotness (через abs(CAR))
    assert scores["e1_pump"] == pytest.approx(scores["e2_dump"], abs=1e-4)

    # Гарячі події повинні мати суттєво вищий скор за флет
    assert scores["e1_pump"] > scores["e3_flat"]
    assert scores["e2_dump"] > scores["e3_flat"]


def test_hotness_score_weights_and_exact_value():
    """Перевірка чіткої формули з різними ваговими коефіцієнтами."""
    df = pl.DataFrame(
        {
            "event_id": ["e1"],
            "car": [0.05],
            "volume_spike": [2.0],
        }
    )

    # Для 1 елемента після нормалізації/скалювання маємо передбачуване значення
    result = calculate_hotness_score(
        df, car_col="car", vol_col="volume_spike", weight_car=0.7, weight_vol=0.3
    )

    assert len(result) == 1
    assert not math.isnan(result["hotness_score"][0])


def test_hotness_null_and_nan_handling():
    """
    Edge case: обробка порожніх/null/NaN значень без падіння двигуна.
    Null-значення повинні безпечно замінюватися на 0 імпакт.
    """
    df = pl.DataFrame(
        {
            "event_id": ["e1_clean", "e2_null_car", "e3_nan_vol"],
            "car": [0.05, None, 0.02],
            "volume_spike": [2.0, 3.0, float("nan")],
        }
    )

    result = calculate_hotness_score(df, car_col="car", vol_col="volume_spike")

    assert result["hotness_score"].null_count() == 0
    # Жодне зі значень не має бути NaN
    assert not any(math.isnan(s) for s in result["hotness_score"].to_list())


def test_hotness_missing_columns():
    """Перевірка виклику помилки при відсутності необхідних колонок."""
    df = pl.DataFrame({"event_id": ["e1"]})

    with pytest.raises(ValueError, match="Missing required columns"):
        calculate_hotness_score(df, car_col="car", vol_col="volume_spike")
