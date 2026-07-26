from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from gram_quant.core.config import settings
from gram_quant.core.schemas import NewsEventRaw
from gram_quant.engine.event_slicer import EventSlicer, EventWindowError
from gram_quant.engine.metrics import MetricsEngine, MetricsError
from gram_quant.fetchers.bybit import BybitFetcher
from gram_quant.fetchers.telegram import TelegramFetcher
from gram_quant.stats.caar import CAAREngine
from gram_quant.visualization.report import EventStudyReport

LOGGER_NAME = "GramEventQuant"
DEFAULT_TELEGRAM_CHANNEL = "durov"
TELEGRAM_MESSAGE_LIMIT = 50
TELEGRAM_SESSION_NAME = "telegram_session"
BYBIT_EVENT_SYMBOL = "TONUSDT"
BYBIT_BASELINE_SYMBOL = "BTCUSDT"
BYBIT_INTERVAL = "1"
PRE_EVENT_MINUTES = 60
POST_EVENT_MINUTES = 120
REPORT_FILE_NAME = "event_study_report.html"
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
            logger.warning("Skipping Telegram row with empty text: %s", row)
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


def build_fetch_window(events: list[tuple[NewsEventRaw, str]]) -> tuple[datetime, datetime]:
    event_times = [event.event_time_kyiv for event, _ in events]
    earliest = min(event_times)
    latest = max(event_times)
    return (
        earliest - timedelta(minutes=PRE_EVENT_MINUTES),
        latest + timedelta(minutes=POST_EVENT_MINUTES),
    )


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
            logger.warning("Skipping event %s due to window error: %s", event_id, error)
            continue

        windows.append(sliced_df.with_columns(pl.lit(event_id).alias("event_id")))

    return pl.concat(windows, how="vertical") if windows else pl.DataFrame()


async def collect_telegram_events() -> list[tuple[NewsEventRaw, str]]:
    if not settings.telegram_api_id or not settings.telegram_api_hash:
        raise ValueError("Telegram API credentials are not configured.")

    fetcher = TelegramFetcher(
        api_id=settings.telegram_api_id,
        api_hash=settings.telegram_api_hash,
        session_name=str(Path(settings.data_raw_dir) / TELEGRAM_SESSION_NAME),
    )

    logger.info("Fetching Telegram messages from %s", DEFAULT_TELEGRAM_CHANNEL)
    messages_df = await fetcher.fetch_channel_messages(
        channel_username=DEFAULT_TELEGRAM_CHANNEL,
        limit=TELEGRAM_MESSAGE_LIMIT,
    )
    logger.info("Fetched %s Telegram messages", messages_df.height)

    return build_news_events(messages_df)


async def fetch_market_data(symbol: str, start_time: datetime, end_time: datetime) -> pl.DataFrame:
    fetcher = BybitFetcher()
    logger.info("Fetching Bybit klines for %s", symbol)
    raw_df = await fetcher.fetch_kline(
        symbol=symbol,
        interval=BYBIT_INTERVAL,
        start_time=start_time,
        end_time=end_time,
    )
    return normalize_ohlcv_timestamp(raw_df)


def build_report(caar_df: pl.DataFrame) -> None:
    report_path = Path(settings.data_processed_dir) / REPORT_FILE_NAME
    reporter = EventStudyReport()
    figure = reporter.build_figure(caar_df, ticker=BYBIT_EVENT_SYMBOL)
    reporter.save_html(figure, str(report_path))
    logger.info("Saved report to %s", report_path)


async def main() -> None:
    logger.info("Starting GramEventQuant analysis pipeline")
    ensure_directories()

    try:
        events = await collect_telegram_events()
    except ValueError as error:
        logger.error("%s", error)
        return

    if not events:
        logger.error("No Telegram events were extracted. Aborting pipeline.")
        return

    fetch_start, fetch_end = build_fetch_window(events)
    ton_df = await fetch_market_data(BYBIT_EVENT_SYMBOL, fetch_start, fetch_end)
    btc_df = await fetch_market_data(BYBIT_BASELINE_SYMBOL, fetch_start, fetch_end)
    logger.info(
        "Retrieved %s TON candles and %s BTC candles",
        ton_df.height,
        btc_df.height,
    )

    slicer = EventSlicer(pre_event_minutes=PRE_EVENT_MINUTES, post_event_minutes=POST_EVENT_MINUTES)
    windowed_df = build_windowed_events(events, ton_df, slicer)
    if windowed_df.is_empty():
        logger.error("No event windows could be constructed. Aborting pipeline.")
        return

    metrics_engine = MetricsEngine()
    try:
        metrics_df = metrics_engine.calculate_metrics(windowed_df)
    except MetricsError as error:
        logger.error("Metrics calculation failed: %s", error)
        return

    caar_engine = CAAREngine()
    caar_df = caar_engine.calculate(metrics_df)
    if caar_df.is_empty():
        logger.error("CAAR result is empty. Aborting pipeline.")
        return

    build_report(caar_df)
    logger.info("GramEventQuant pipeline completed successfully")


if __name__ == "__main__":
    asyncio.run(main())
