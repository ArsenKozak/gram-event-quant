from pathlib import Path

import polars as pl

from gram_quant.core.config import settings
from gram_quant.visualization.report import EventVisualizer


DASHBOARD_FILE_NAME = "hotness_dashboard.html"


def main():
    input_path = Path(settings.data_processed_dir) / "events_merged_with_candles.csv"
    output_path = Path(settings.data_processed_dir) / DASHBOARD_FILE_NAME

    print(f"Loading events from {input_path}")

    events_df = pl.read_csv(input_path)

    print(f"Loaded {len(events_df)} events")

    visualizer = EventVisualizer()

    visualizer.export_html_report(
        events_df,
        output_path=str(output_path),
    )

    print(f"Saved dashboard: {output_path}")


if __name__ == "__main__":
    main()