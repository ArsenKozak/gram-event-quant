import json
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots


class EventStudyReport:
    """Генератор візуальних звітів та HTML-дашбордів для Event Study на базі Plotly."""

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

        x_vals = caar_df["relative_minute"].to_list() if not caar_df.is_empty() else []

        # 1. Mean CAAR Line
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=caar_df["mean_return"].to_list() if not caar_df.is_empty() else [],
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
                y=caar_df["median_return"].to_list() if not caar_df.is_empty() else [],
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
                y=caar_df["mean_volume_spike"].to_list() if not caar_df.is_empty() else [],
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

    def build_individual_car_figure(
        self, metrics_df: pl.DataFrame, caar_df: pl.DataFrame, ticker: str = "GRAMUSDT"
    ) -> go.Figure:
        """
        Рендерить графік індивідуальних CAR для кожної події разом з усередненим CAAR.
        Використовується для дебагу стабільності сигналу (пошуку аномальних викидів).
        """
        fig = go.Figure()

        if not metrics_df.is_empty():
            # Розбиваємо датафрейм на окремі партиції (ізольовані треди подій)
            events = metrics_df.partition_by("event_id")

            for event_data in events:
                event_data = event_data.sort("relative_minute")
                event_id = event_data["event_id"][0]

                # Додаємо графік кожного CAR як напівпрозору лінію
                fig.add_trace(
                    go.Scatter(
                        x=event_data["relative_minute"].to_list(),
                        y=event_data["car"].to_list(),
                        mode="lines",
                        name=f"Event {event_id}",
                        line={"width": 1, "color": "rgba(150, 150, 150, 0.3)"},
                        showlegend=False,
                        hoverinfo="text",
                        hovertext=f"ID: {event_id}",
                    )
                )

        # Додаємо головний агрегований процес (CAAR) як масивну лінію поверх усіх
        if not caar_df.is_empty():
            fig.add_trace(
                go.Scatter(
                    x=caar_df["relative_minute"].to_list(),
                    y=caar_df["mean_return"].to_list(),
                    mode="lines",
                    name="Mean CAAR",
                    line={"color": "#1f77b4", "width": 4},
                )
            )

        # Маркер часу нуль (виклик тригера)
        fig.add_vline(
            x=0,
            line_width=2,
            line_dash="dot",
            line_color="red",
            annotation_text="T0 (Trigger)",
            annotation_position="top left",
        )

        fig.update_layout(
            title=f"Individual Event CARs vs Mean CAAR — {ticker}",
            xaxis_title="Relative Minute",
            yaxis_title="Cumulative Abnormal Return (CAR)",
            template="plotly_white",
            height=700,
            showlegend=True,
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )

        return fig

    def export_html_report(
        self,
        df_events: pl.DataFrame,
        output_path: str = "reports/dashboard.html",
    ) -> str:
        """
        Експортує DataFrame з подіями, Hotness Index та деталями в автономний HTML-дашборд.
        Гарантує наявність строгого JSON payload для контракту та валідації.
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        is_empty = df_events.is_empty()
        events_data = df_events.to_dicts() if not is_empty else []

        payload = {
            "is_empty": is_empty,
            "count": len(events_data),
            "events": events_data,
        }

        json_payload_str = json.dumps(payload, ensure_ascii=False, indent=2, default=str)

        table_rows = ""
        for e in events_data:
            table_rows += f"""
                <tr>
                    <td>{e.get("event_id", "")}</td>
                    <td><span class="hotness-badge">{e.get("hotness_score", 0.0):.2f}</span></td>
                    <td>{e.get("car", 0.0):.4f}</td>
                    <td>{e.get("volume_spike", 0.0):.2f}x</td>
                    <td>{e.get("post_text", "")}</td>
                </tr>"""

        html_content = f"""<!DOCTYPE html>
<html lang="uk">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Gram Event Quant Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
        h1 {{ color: #38bdf8; text-align: center; }}
        .card {{ background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1); }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background-color: #0f172a; color: #94a3b8; }}
        tr:hover {{ background-color: #334155; cursor: pointer; }}
        .hotness-badge {{ background: #f97316; color: white; padding: 4px 8px; border-radius: 4px; font-weight: bold; }}
        .empty-state {{ text-align: center; color: #94a3b8; padding: 40px; }}
    </style>
</head>
<body>
    <h1>🔥 Hotness Index & Event Dashboard</h1>
    
    <div class="card">
        <h2>Event Analytics Overview</h2>
        {
            "<div class='empty-state'>Немає даних для відображення</div>"
            if is_empty
            else f'''
        <table>
            <thead>
                <tr>
                    <th>Event ID</th>
                    <th>Hotness Score</th>
                    <th>CAR</th>
                    <th>Volume Spike</th>
                    <th>Post Text</th>
                </tr>
            </thead>
            <tbody>{table_rows}
            </tbody>
        </table>
        '''
        }
    </div>

    <!-- Embedded Structured Data Payload for Contract Testing & Client Hydration -->
    <script id="quant-data" type="application/json">
{json_payload_str}
    </script>
</body>
</html>
"""

        path.write_text(html_content, encoding="utf-8")
        return str(path)


# Аліас для сумісності з імпортами visualizer
EventVisualizer = EventStudyReport
