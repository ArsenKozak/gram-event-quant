import plotly.graph_objects as go
import polars as pl
import pytest

from gram_quant.visualization.report import EventStudyReport


@pytest.fixture
def sample_caar_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "relative_minute": [-2, -1, 0, 1, 2],
            "mean_return": [-0.001, 0.0, 0.015, 0.025, 0.022],
            "median_return": [-0.001, 0.0, 0.012, 0.020, 0.021],
            "std_return": [0.005, 0.004, 0.010, 0.012, 0.011],
            "mean_volume_spike": [1.0, 1.1, 4.5, 3.2, 1.8],
            "event_count": [10, 10, 10, 10, 10],
        }
    )


def test_generate_plotly_figure(sample_caar_df):
    """Перевірка створення об'єкта Plotly Figure."""
    reporter = EventStudyReport()
    fig = reporter.build_figure(sample_caar_df, ticker="GRAMUSDT")

    assert isinstance(fig, go.Figure)
    # Має бути 3 графіки: Mean CAAR, Median CAAR, Volume
    assert len(fig.data) >= 3


def test_export_html_report(sample_caar_df, tmp_path):
    """Перевірка експорту звіту в HTML файл."""
    reporter = EventStudyReport()
    fig = reporter.build_figure(sample_caar_df, ticker="GRAMUSDT")

    output_file = tmp_path / "test_report.html"
    reporter.save_html(fig, str(output_file))

    assert output_file.exists()
    assert output_file.stat().st_size > 0
