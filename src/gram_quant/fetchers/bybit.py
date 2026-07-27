import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
import polars as pl
from loguru import logger


class BybitFetcher:
    """
    Async Bybit V5 OHLCV fetcher.
    Public API, no keys required.
    """

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout

    async def fetch_synthetic_kline(
            self,
            symbol: str = "GRAMUSDT",
            category: str = "spot",
            interval: str = "1",
            start_time: datetime | None = None,
            end_time: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Backward compatibility wrapper.
        """
        return await self.fetch_kline(
            symbol=symbol,
            category=category,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
        )

    async def _request_with_retry(
            self,
            client: httpx.AsyncClient,
            params: dict[str, str | int],
            retries: int = 4,
    ) -> dict[str, Any]:

        for attempt in range(retries):
            try:
                response = await client.get(
                    self.BASE_URL,
                    params=params,
                )

                response.raise_for_status()
                return response.json()

            except (
                    httpx.ReadError,
                    httpx.ConnectError,
                    httpx.TimeoutException,
            ) as error:

                if attempt == retries - 1:
                    raise

                delay = 2 ** attempt

                logger.warning(
                    "Bybit network error: %s. Retry %s/%s in %ss",
                    error,
                    attempt + 1,
                    retries,
                    delay,
                    )

                await asyncio.sleep(delay)

        raise RuntimeError("Unreachable")

    async def fetch_kline(
            self,
            symbol: str = "BTCUSDT",
            category: str = "spot",
            interval: str = "1",
            start_time: datetime | None = None,
            end_time: datetime | None = None,
            limit: int = 1000,
    ) -> pl.DataFrame:

        clean_symbol = symbol.upper().strip()

        params_base: dict[str, str | int] = {
            "category": category,
            "symbol": clean_symbol,
            "interval": interval,
            "limit": limit,
        }

        start_ms = (
            int(start_time.astimezone(UTC).timestamp() * 1000)
            if start_time
            else None
        )

        end_ms = (
            int(end_time.astimezone(UTC).timestamp() * 1000)
            if end_time
            else None
        )

        all_records: list[list[Any]] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:

            logger.info(
                "Fetching %s %s candles from %s to %s",
                clean_symbol,
                category,
                start_time,
                end_time,
            )

            current_end = (
                end_ms
                if end_ms is not None
                else int(datetime.now(UTC).timestamp() * 1000)
            )

            page = 0

            while True:

                page += 1

                params = {
                    **params_base,
                    "end": current_end,
                }

                data = await self._request_with_retry(
                    client,
                    params,
                )

                if data.get("retCode") != 0:
                    raise ValueError(
                        f"Bybit API Error ({clean_symbol}): "
                        f"{data.get('retMsg')}"
                    )

                raw_list = (
                    data.get("result", {})
                    .get("list", [])
                )

                if not raw_list:
                    logger.info(
                        "No more data page=%s symbol=%s",
                        page,
                        clean_symbol,
                    )
                    break

                all_records.extend(raw_list)

                timestamps = [
                    int(item[0])
                    for item in raw_list
                ]

                min_ts = min(timestamps)

                logger.info(
                    "Page %s collected %s candles",
                    page,
                    len(raw_list),
                )

                if start_ms and min_ts <= start_ms:
                    break

                if len(raw_list) < limit:
                    break

                current_end = min_ts - 1

                # protect Bybit API
                await asyncio.sleep(0.2)


        if not all_records:
            logger.warning(
                "No kline data collected for %s",
                clean_symbol,
            )
            return pl.DataFrame()


        unique = {}

        for item in all_records:
            try:
                unique[int(item[0])] = item
            except Exception:
                continue


        records = [
            {
                "timestamp": int(item[0]),
                "open": float(item[1]),
                "high": float(item[2]),
                "low": float(item[3]),
                "close": float(item[4]),
                "volume": float(item[5]),
                "turnover": float(item[6]),
            }
            for item in unique.values()
        ]


        df = (
            pl.DataFrame(records)
            .with_columns(
                pl.from_epoch(
                    "timestamp",
                    time_unit="ms",
                )
                .dt.replace_time_zone("UTC")
                .alias("datetime")
            )
            .sort("datetime")
        )


        logger.success(
            "Fetched %s candles for %s",
            len(df),
            clean_symbol,
        )

        return df