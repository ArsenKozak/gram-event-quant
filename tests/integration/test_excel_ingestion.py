from pathlib import Path

import pytest

from gram_quant.engine.ingestion import ExcelIngestor

FIXTURE_PATH = Path("tests/fixtures/sample_events.xlsx")


def test_excel_ingestor_load_dataframe():
    """Тестує реальне зчитування Excel файлу через Polars."""
    ingestor = ExcelIngestor(FIXTURE_PATH)
    df = ingestor.load_raw_dataframe()

    assert not df.is_empty()
    assert len(df) == 3
    assert "author" in df.columns


def test_excel_ingestor_parse_and_clean_pipeline():
    """
    Тестує повний пайплайн обробки:
    - Зчитування
    - Конвертація часу
    - Пропуск битих рядків (збереження лише валідних)
    """
    ingestor = ExcelIngestor(FIXTURE_PATH)
    df = ingestor.load_raw_dataframe()
    events = ingestor.parse_and_clean(df)

    # З 3 рядків фікстури лише 2 валідні (третій має невалідну важливість/ціну)
    assert len(events) == 2
    assert events[0].author == "Pavel Durov"
    assert events[0].price_before == pytest.approx(0.015)
    assert events[0].event_time_kyiv.hour == 15
