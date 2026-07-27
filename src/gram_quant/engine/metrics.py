from typing import Tuple
import polars as pl

from gram_quant.stats.market_model import calculate_market_model_ar


class MetricsError(ValueError):
    """Викликається при критичних помилках розрахунку метрик."""

    pass


class MetricsEngine:
    """Двигун для розрахунку кількісних метрик (Returns, Volume Spikes, Market-Adjusted AR)."""

    def calculate_metrics(self, df: pl.DataFrame) -> pl.DataFrame:
        if df.is_empty():
            return df

        baseline_check = df.group_by("event_id").agg(
            pl.col("relative_minute").eq(-1).sum().alias("baseline_count")
        )

        if baseline_check["baseline_count"].min() == 0:
            raise MetricsError("Missing baseline: event without relative_minute == -1")
        if baseline_check["baseline_count"].max() > 1:
            raise MetricsError("Duplicate baseline: event with multiple relative_minute == -1")

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

    def calculate_market_adjusted_metrics(
            self,
            estimation_df: pl.DataFrame,
            event_df: pl.DataFrame,
            asset_col: str = "returns_asset",
            market_col: str = "returns_market",
            event_time_col: str = "relative_minute",
    ) -> Tuple[pl.DataFrame, float, float]:
        """
        Обчислює Clean AR та CAR на основі OLS-регресії проти ринку (BTC).

        Returns:
            Tuple[pl.DataFrame, float, float]: (event_df z 'abnormal_return' i 'car', alpha, beta)
        """
        # Перевірка наявності необхідних колонок
        for df, name in [(estimation_df, "estimation_df"), (event_df, "event_df")]:
            missing = [col for col in [asset_col, market_col] if col not in df.columns]
            if missing:
                raise MetricsError(f"Missing required columns {missing} in {name}")

        # Розрахунок OLS-регресії та Abnormal Returns через модуль market_model
        result_df, alpha, beta = calculate_market_model_ar(
            estimation_df=estimation_df,
            event_df=event_df,
            asset_col=asset_col,
            market_col=market_col,
        )

        # Сортування та розрахунок Cumulative Abnormal Return (CAR)
        if event_time_col in result_df.columns:
            result_df = result_df.sort(event_time_col)

        result_df = result_df.with_columns(
            pl.col("abnormal_return").cum_sum().alias("car")
        )

        return result_df, alpha, beta