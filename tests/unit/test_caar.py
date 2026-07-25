import polars as pl
import pytest

from gram_quant.stats.caar import CAAREngine


@pytest.fixture
def multi_event_metrics() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_id": ["e1", "e1", "e2", "e2"],
            "relative_minute": [0, 1, 0, 1],
            "cumulative_return": [0.02, 0.04, 0.04, 0.08],
            "volume_spike": [1.0, 1.5, 3.0, 2.5],
        }
    )


def test_caar_basic_calculation_and_sorting(multi_event_metrics):
    """Базова перевірка математики середнього та сортування по часу."""
    engine = CAAREngine()
    df = engine.calculate(multi_event_metrics)

    assert len(df) == 2
    assert df["relative_minute"].is_sorted()

    t0_row = df.filter(pl.col("relative_minute") == 0)
    assert t0_row["mean_return"][0] == pytest.approx(0.03)  # (0.02 + 0.04) / 2
    assert t0_row["event_count"][0] == 2


def test_caar_median_with_outliers():
    """Перевірка стійкості медіани до екстремальних викидів (outliers)."""
    df = pl.DataFrame(
        {
            "event_id": ["e1", "e2", "e3"],
            "relative_minute": [0, 0, 0],
            "cumulative_return": [0.02, 0.04, 10.0],  # 10.0 - це аномальний викид
            "volume_spike": [1.0, 1.0, 1.0],
        }
    )

    engine = CAAREngine()
    res = engine.calculate(df)

    # Середнє буде викривлене (3.35), але медіана має залишитися 0.04
    assert res["median_return"][0] == pytest.approx(0.04)


def test_caar_empty_dataframe():
    """Порожній датафрейм не повинен ламати рушій."""
    engine = CAAREngine()
    res = engine.calculate(pl.DataFrame())
    assert res.is_empty()


def test_caar_missing_values():
    """Null/None значення не повинні ламати агрегацію."""
    df = pl.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "relative_minute": [0, 0],
            "cumulative_return": [0.05, None],
            "volume_spike": [1.0, 2.0],
        }
    )

    engine = CAAREngine()
    res = engine.calculate(df)

    # Має проігнорувати None і порахувати середнє по існуючих
    assert res["mean_return"][0] == pytest.approx(0.05)
    assert res["event_count"][0] == 1  # Валідних івентів для цієї метрики лише 1


def test_caar_duplicate_event_minutes():
    """Два записи для одного івенту в ту саму хвилину мають викликати помилку."""
    df = pl.DataFrame(
        {
            "event_id": ["e1", "e1"],
            "relative_minute": [0, 0],
            "cumulative_return": [0.02, 0.04],
            "volume_spike": [1.0, 1.0],
        }
    )

    engine = CAAREngine()
    with pytest.raises(ValueError, match="Duplicate relative_minute for the same event_id"):
        engine.calculate(df)
