from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from gram_quant.core.config import settings
from gram_quant.core.schemas import NewsEventRaw
from gram_quant.engine.event_slicer import EventSlicer, EventWindowError
from gram_quant.engine.hotness import calculate_hotness_score
from gram_quant.engine.metrics import MetricsEngine, MetricsError
from gram_quant.fetchers.bybit import BybitFetcher
from gram_quant.fetchers.telegram import TelegramFetcher
from gram_quant.stats.caar import CAAREngine
from gram_quant.storage.duckdb_store import DuckDBStore
from gram_quant.visualization.report import EventStudyReport, EventVisualizer

LOGGER_NAME = "GramEventQuant"
DEFAULT_TELEGRAM_CHANNEL = "durov"
TELEGRAM_MESSAGE_LIMIT = None
TELEGRAM_SESSION_NAME = "telegram_session"
BYBIT_EVENT_SYMBOL = "GRAMUSDT"
BYBIT_BASELINE_SYMBOL = "BTCUSDT"
BYBIT_INTERVAL = "1"

# Фіксований фрейм для бектестингу гіпотези
TELEGRAM_FETCH_START_DATE = datetime(2023, 1, 1, tzinfo=UTC)
BYBIT_FETCH_START_DATE = datetime(2023, 1, 1, tzinfo=UTC)

PRE_EVENT_MINUTES = 60
POST_EVENT_MINUTES = 120
REPORT_FILE_NAME = "event_study_report.html"
DASHBOARD_FILE_NAME = "dashboard.html"
DEFAULT_IMPORTANCE = 3
DEFAULT_PRICE_BEFORE = 1.0
DEFAULT_BTC_BEFORE = 1.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(LOGGER_NAME)


def ensure_directories() -> None:
    Path(settings.data_raw_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.data_processed_dir).mkdir(parents=True, exist_ok=True)


def normalize_ohlcv_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    if df.is_empty():
        return df

    if "timestamp" not in df.columns:
        raise ValueError("OHLCV dataframe must contain a 'timestamp' column")

    if df["timestamp"].dtype == pl.Int64:
        return df.with_columns(
            pl.from_epoch("timestamp", time_unit="ms")
            .dt.replace_time_zone("UTC")
            .alias("timestamp")
        ).sort("timestamp")

    return df.with_columns(pl.col("timestamp").dt.replace_time_zone("UTC").alias("timestamp"))


def build_event_id(channel: str, msg_id: int | str) -> str:
    return f"{channel}:{msg_id}"


def build_news_events(messages_df: pl.DataFrame) -> list[tuple[NewsEventRaw, str]]:
    events: list[tuple[NewsEventRaw, str]] = []

    for row_index, row in enumerate(messages_df.iter_rows(named=True), start=1):
        timestamp = row.get("datetime")
        if not isinstance(timestamp, datetime):
            logger.warning("Skipping Telegram row without valid datetime: %s", row)
            continue

        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)

        channel = str(row.get("channel") or DEFAULT_TELEGRAM_CHANNEL)
        text = str(row.get("text") or "").strip()

        if not text:
            continue

        msg_id = row.get("msg_id") or row_index
        event_id = build_event_id(channel, msg_id)

        event = NewsEventRaw(
            event_time_kyiv=timestamp,
            author=channel,
            url=None,
            text=text,
            importance=DEFAULT_IMPORTANCE,
            price_before=DEFAULT_PRICE_BEFORE,
            btc_before=DEFAULT_BTC_BEFORE,
        )
        events.append((event, event_id))

    return events


def build_events_dataframe(events: list[tuple[NewsEventRaw, str]]) -> pl.DataFrame:
    records: list[dict[str, str | int | datetime | float]] = []

    for event, event_id in events:
        msg_id_part = event_id.split(":", 1)[1]
        try:
            msg_id = int(msg_id_part)
        except ValueError:
            msg_id = msg_id_part

        event_time = event.event_time_kyiv
        if event_time.tzinfo is None:
            event_time = event_time.replace(tzinfo=UTC)

        records.append(
            {
                "event_id": event_id,
                "msg_id": msg_id,
                "datetime": event_time,
                "channel": event.author,
                "text": event.text,
                "importance": event.importance,
                "price_before": event.price_before,
                "btc_before": event.btc_before,
            }
        )

    return pl.DataFrame(records)


def build_windowed_events(
    events: list[tuple[NewsEventRaw, str]],
    ohlcv_df: pl.DataFrame,
    slicer: EventSlicer,
) -> pl.DataFrame:
    windows: list[pl.DataFrame] = []

    for event, event_id in events:
        try:
            sliced_df = slicer.slice_window(event, ohlcv_df)
        except EventWindowError as error:
            logger.debug("Skipping event %s due to window error: %s", event_id, error)
            continue

        windows.append(sliced_df.with_columns(pl.lit(event_id).alias("event_id")))

    return pl.concat(windows, how="vertical") if windows else pl.DataFrame()


def safe_write_csv(df: pl.DataFrame, path: Path) -> None:
    try:
        df.write_csv(path)
    except Exception as error:
        logger.warning("Failed to save %s: %s", path, error)
    else:
        logger.info("Saved CSV to %s", path)


def safe_write_parquet(df: pl.DataFrame, path: Path) -> None:
    try:
        df.write_parquet(path)
    except Exception as error:
        logger.warning("Failed to save %s: %s", path, error)
    else:
        logger.info("Saved Parquet to %s", path)


def export_full_market_data_duckdb(ton_df: pl.DataFrame, btc_df: pl.DataFrame) -> None:
    datasets = []

    for symbol, dataframe in ((BYBIT_EVENT_SYMBOL, ton_df), (BYBIT_BASELINE_SYMBOL, btc_df)):
        if dataframe.is_empty():
            continue
        datasets.append(
            dataframe.with_columns(
                pl.lit(symbol).alias("symbol"),
                pl.col("timestamp")
                .dt.replace_time_zone("UTC")
                .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
                .alias("timestamp_str"),
            )
        )

    if not datasets:
        return

    raw_ohlcv_full = pl.concat(datasets, how="vertical")

    try:
        raw_dir = Path(settings.data_raw_dir)
        raw_dir.mkdir(parents=True, exist_ok=True)

        parquet_path = raw_dir / "ohlcv_full.parquet"
        safe_write_parquet(raw_ohlcv_full, parquet_path)

        with DuckDBStore(db_path=str(raw_dir / "gram_quant.duckdb")) as store:
            store.conn.execute(
                f"CREATE OR REPLACE TABLE ohlcv_raw AS SELECT * FROM read_parquet('{parquet_path}')"
            )
    except Exception as error:
        logger.warning("Failed to store market data in DuckDB: %s", error)


async def collect_telegram_events() -> tuple[list[tuple[NewsEventRaw, str]], pl.DataFrame]:
    if not settings.telegram_api_id or not settings.telegram_api_hash:
        raise ValueError("Telegram API credentials are not configured.")

    fetcher = TelegramFetcher(
        api_id=settings.telegram_api_id,
        api_hash=settings.telegram_api_hash,
        session_name=str(Path(settings.data_raw_dir) / TELEGRAM_SESSION_NAME),
    )

    messages_df = await fetcher.fetch_channel_messages(
        channel_username=DEFAULT_TELEGRAM_CHANNEL,
        limit=TELEGRAM_MESSAGE_LIMIT,
        start_date=TELEGRAM_FETCH_START_DATE,
    )
    return build_news_events(messages_df), messages_df


async def fetch_market_data(symbol: str, start_time: datetime, end_time: datetime) -> pl.DataFrame:
    fetcher = BybitFetcher()
    # Використовуємо synthetic метод для зшивання TON та GRAM
    raw_df = await fetcher.fetch_synthetic_kline(
        symbol=symbol,
        interval=BYBIT_INTERVAL,
        start_time=start_time,
        end_time=end_time,
    )
    return normalize_ohlcv_timestamp(raw_df)


async def main() -> None:
    logger.info("Starting GramEventQuant analysis pipeline")
    ensure_directories()

    try:
        events, messages_df = await collect_telegram_events()
    except ValueError as error:
        logger.error("%s", error)
        return

    if not events:
        logger.error("No Telegram events were extracted.")
        return

    events_df = build_events_dataframe(events)
    fetch_start = BYBIT_FETCH_START_DATE
    fetch_end = datetime.now(UTC)

    ton_df = await fetch_market_data(BYBIT_EVENT_SYMBOL, fetch_start, fetch_end)
    btc_df = await fetch_market_data(BYBIT_BASELINE_SYMBOL, fetch_start, fetch_end)

    export_full_market_data_duckdb(ton_df, btc_df)
    safe_write_csv(messages_df, Path(settings.data_processed_dir) / "telegram_messages.csv")

    # Синхронізація бенчмарку (BTC) в основний DataFrame для регресії
    btc_subset = btc_df.select([pl.col("timestamp"), pl.col("close").alias("close_btc")])
    global_df = ton_df.join(btc_subset, on="timestamp", how="left").with_columns(
        pl.col("close_btc").fill_null(strategy="forward")
    )

    slicer = EventSlicer(pre_event_minutes=PRE_EVENT_MINUTES, post_event_minutes=POST_EVENT_MINUTES)
    windowed_df = build_windowed_events(events, global_df, slicer)
    if windowed_df.is_empty():
        logger.error("No event windows could be constructed.")
        return

    # Розрахунок returns ізольовано для кожного івенту
    windowed_df = windowed_df.with_columns(
        returns_asset=pl.col("close").pct_change().over("event_id").fill_null(0.0),
        returns_market=pl.col("close_btc").pct_change().over("event_id").fill_null(0.0),
    )

    metrics_engine = MetricsEngine()
    try:
        metrics_df = metrics_engine.calculate_metrics(windowed_df)

        # Підготовка глобального estimation window для OLS
        est_df = global_df.with_columns(
            returns_asset=pl.col("close").pct_change().fill_null(0.0),
            returns_market=pl.col("close_btc").pct_change().fill_null(0.0),
        ).drop_nulls()

        # Розрахунок Market-Adjusted AR
        metrics_df, alpha, beta = metrics_engine.calculate_market_adjusted_metrics(
            estimation_df=est_df,
            event_df=metrics_df,
            asset_col="returns_asset",
            market_col="returns_market",
        )

        # Ізоляція кумулятивної суми по івентах (фікс state-leakage)
        metrics_df = metrics_df.with_columns(
            pl.col("abnormal_return").cum_sum().over("event_id").alias("car")
        )
    except MetricsError as error:
        logger.error("Metrics calculation failed: %s", error)
        return

    safe_write_csv(metrics_df, Path(settings.data_processed_dir) / "events_merged_with_candles.csv")

    caar_engine = CAAREngine()
    caar_df = caar_engine.calculate(metrics_df)

    # 1. Експорт основного графіку CAAR
    report_path = Path(settings.data_processed_dir) / REPORT_FILE_NAME
    reporter = EventStudyReport()
    figure = reporter.build_figure(caar_df, ticker=BYBIT_EVENT_SYMBOL)
    reporter.save_html(figure, str(report_path))
    logger.info("Saved Plotly CAAR report to %s", report_path)

    # 2. Експорт графіка розсіювання (Individual CARs)
    individual_fig = reporter.build_individual_car_figure(
        metrics_df=metrics_df, caar_df=caar_df, ticker=BYBIT_EVENT_SYMBOL
    )
    individual_report_path = Path(settings.data_processed_dir) / "individual_events_report.html"
    reporter.save_html(individual_fig, str(individual_report_path))
    logger.info("Saved Plotly Individual Events report to %s", individual_report_path)

    # 3. Агрегація Hotness Score та експорт JSON-гідрованого дашборду
    event_agg = metrics_df.group_by("event_id").agg(
        pl.col("car").last().alias("car"), pl.col("volume_spike").max().alias("volume_spike")
    )
    event_agg = calculate_hotness_score(event_agg, car_col="car", vol_col="volume_spike")

    events_final_df = events_df.join(event_agg, on="event_id", how="left").with_columns(
        pl.col("text").alias("post_text")
    )

    dash_path = Path(settings.data_processed_dir) / DASHBOARD_FILE_NAME
    visualizer = EventVisualizer()
    visualizer.export_html_report(events_final_df, output_path=str(dash_path))
    logger.info("Saved Interactive Hotness Dashboard to %s", dash_path)
    logger.info("GramEventQuant pipeline completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
