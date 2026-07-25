import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


class EventStudyReport:
    """Генератор візуальних звітів для Event Study на базі Plotly."""

    def build_figure(self, caar_df: pl.DataFrame, ticker: str = "GRAMUSDT") -> go.Figure:
        """Створює двоосьовий інтерактивний графік CAAR та Volume Spike."""
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.08,
            subplot_titles=(
                f"Cumulative Average Abnormal Return (CAAR) — {ticker}",
                "Average Volume Spike (Relative to Baseline)",
            ),
            row_heights=[0.7, 0.3],
        )

        x_vals = caar_df["relative_minute"].to_list()

        # 1. Mean CAAR Line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=caar_df["mean_return"].to_list(),
                mode="lines+markers",
                name="Mean CAAR",
                line={"color": "#1f77b4", "width": 2.5},
            ),
            row=1,
            col=1,
        )

        # 2. Median CAAR Line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=caar_df["median_return"].to_list(),
                mode="lines",
                name="Median CAAR",
                line={"color": "#ff7f0e", "width": 2, "dash": "dash"},
            ),
            row=1,
            col=1,
        )

        # 3. Volume Spike Bars
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=caar_df["mean_volume_spike"].to_list(),
                name="Mean Volume Spike",
                marker_color="#2ca02c",
                opacity=0.75,
            ),
            row=2,
            col=1,
        )

        # Додаємо лінію T0
        for row in [1, 2]:
            fig.add_vline(
                x=0,
                line_width=2,
                line_dash="dot",
                line_color="red",
                annotation_text="T0 (Post Published)" if row == 1 else "",
                annotation_position="top left",
                row=row,
                col=1,
            )

        # Стилізація
        fig.update_layout(
            template="plotly_white",
            height=700,
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
            margin={"l": 50, "r": 50, "t": 80, "b": 50},
        )

        fig.update_yaxes(title_text="Return (Decimal)", row=1, col=1)
        fig.update_yaxes(title_text="Volume Multiplier", row=2, col=1)
        fig.update_xaxes(title_text="Relative Minute (T0 = Post)", row=2, col=1)

        return fig

    def save_html(self, fig: go.Figure, output_path: str) -> None:
        """Зберігає графік у закритий автономний HTML-файл."""
        fig.write_html(output_path, include_plotlyjs="cdn")
