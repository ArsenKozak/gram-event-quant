import polars as pl


class CAAREngine:
    """Двигун для розрахунку Cumulative Average Abnormal Return (CAAR) по масиву подій."""

    def calculate(self, metrics_df: pl.DataFrame) -> pl.DataFrame:
        """Агрегує метрики всіх івентів по хвилинах відносно T0."""
        if metrics_df.is_empty():
            return metrics_df

        # 1. Валідація: не повинно бути дублікатів хвилин для одного івенту
        if not metrics_df.select(["event_id", "relative_minute"]).is_unique().all():
            raise ValueError("Duplicate relative_minute for the same event_id")

        # 2. Агрегація метрик по кожній хвилині
        caar_df = (
            metrics_df.group_by("relative_minute")
            .agg(
                # Кількість івентів, для яких є валідна прибутковість у цю хвилину
                pl.col("cumulative_return").drop_nulls().len().alias("event_count"),
                # Основні статистики прибутковості
                pl.col("cumulative_return").mean().alias("mean_return"),
                pl.col("cumulative_return").median().alias("median_return"),
                pl.col("cumulative_return").std().alias("std_return"),
                # Статистики об'єму
                pl.col("volume_spike").mean().alias("mean_volume_spike"),
                pl.col("volume_spike").median().alias("median_volume_spike"),
            )
            .sort("relative_minute")
        )

        return caar_df
