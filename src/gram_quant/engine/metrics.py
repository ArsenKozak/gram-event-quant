import polars as pl


class MetricsError(ValueError):
    """Викликається при критичних помилках розрахунку метрик."""
    pass


class MetricsEngine:
    """Двигун для розрахунку кількісних метрик (Returns, Volume Spikes)."""

    def calculate_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        baseline_check = df.group_by("event_id").agg(
            pl.col("relative_minute").eq(-1).sum().alias("baseline_count")
        )

        if baseline_check["baseline_count"].min() == 0:
            raise MetricsError("Missing baseline: event without relative_minute == -1")
        if baseline_check["baseline_count"].max() > 1:
            raise MetricsError(
                "Duplicate baseline: event with multiple relative_minute == -1"
            )

        df_metrics = df.with_columns(
            base_close=pl.col("close")
            .filter(pl.col("relative_minute") == -1)
            .first()
            .over("event_id"),
            base_volume=pl.col("volume")
            .filter(pl.col("relative_minute") < 0)
            .mean()
            .over("event_id"),
        )

        df_metrics = df_metrics.with_columns(
            cumulative_return=(pl.col("close") - pl.col("base_close")) / pl.col("base_close"),
            volume_spike=pl.when(pl.col("base_volume") > 0)
            .then(pl.col("volume") / pl.col("base_volume"))
            .otherwise(None),
        )

        return df_metrics.drop(["base_close", "base_volume"])
