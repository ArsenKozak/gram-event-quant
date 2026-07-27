import json
import re

import polars as pl
import pytest

from gram_quant.visualization.visualizer import EventVisualizer


def _extract_json_payload(html_content: str) -> dict:
    """Допоміжна функція для витягування структурованого JSON з HTML-звіту."""
    # Шукаємо скрипт з ID або regex паттерном для JSON payload
    match = re.search(
        r'<script\s+id=["\']quant-data["\']\s+type=["\']application/json["\']>(.*?)</script>',
        html_content,
        re.DOTALL,
    )
    if not match:
        raise ValueError("Quant JSON payload script tag missing in HTML report")
    return json.loads(match.group(1))


def test_export_html_report_data_payload_contract(tmp_path):
    """
    10/10 Contract Test:
    Парсить зашитий JSON payload усередині HTML і перевіряє точні математичні значення,
    не залежачи від зміни CSS, класів чи верстки UI.
    """
    df_events = pl.DataFrame(
        {
            "event_id": ["e1", "e2"],
            "relative_minute": [0, 0],
            "close": [1.0, 2.0],
            "car": [0.05, -0.08],
            "volume_spike": [2.5, 4.0],
            "hotness_score": [0.65, 0.90],
            "post_text": ["TON Announcement!", "Major Token Burn"],
        }
    )

    output_file = tmp_path / "dashboard_test.html"
    visualizer = EventVisualizer()
    visualizer.export_html_report(df_events, output_path=str(output_file))

    assert output_file.exists()
    content = output_file.read_text(encoding="utf-8")

    # 1. Структурна валідація HTML DOM
    assert "<!DOCTYPE html>" in content
    assert "</html>" in content

    # 2. Парсинг та перевірка ізольованого JSON Payload (Жодної крихкості від текстових назв)
    payload = _extract_json_payload(content)

    assert len(payload["events"]) == 2

    e1 = payload["events"][0]
    e2 = payload["events"][1]

    # Точні перевірки метрик без "0.90 in content"
    assert e1["event_id"] == "e1"
    assert e1["car"] == pytest.approx(0.05)
    assert e1["hotness_score"] == pytest.approx(0.65)
    assert e1["post_text"] == "TON Announcement!"

    assert e2["event_id"] == "e2"
    assert e2["car"] == pytest.approx(-0.08)
    assert e2["hotness_score"] == pytest.approx(0.90)
    assert e2["post_text"] == "Major Token Burn"


def test_export_html_report_empty_dataframe_contract(tmp_path):
    """
    Контрактний тест порожнього звітного пайплайну:
    Гарантує, що payload містить порожній масив та прапорець is_empty=True.
    """
    df_empty = pl.DataFrame(
        {
            "event_id": pl.Series([], dtype=pl.Utf8),
            "relative_minute": pl.Series([], dtype=pl.Int64),
            "close": pl.Series([], dtype=pl.Float64),
            "car": pl.Series([], dtype=pl.Float64),
            "volume_spike": pl.Series([], dtype=pl.Float64),
            "hotness_score": pl.Series([], dtype=pl.Float64),
            "post_text": pl.Series([], dtype=pl.Utf8),
        }
    )

    output_file = tmp_path / "dashboard_empty.html"
    visualizer = EventVisualizer()
    visualizer.export_html_report(df_empty, output_path=str(output_file))

    content = output_file.read_text(encoding="utf-8")
    payload = _extract_json_payload(content)

    assert payload["is_empty"] is True
    assert len(payload["events"]) == 0
