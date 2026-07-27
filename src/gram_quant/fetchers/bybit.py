import asyncio
from datetime import UTC, datetime
from typing import Any

import httpx
import polars as pl
from loguru import logger


class BybitFetcher:
    """
    Асинхронний фетчер для завантаження публічних OHLCV свічок з Bybit V5 API.
    Працює без реєстрації та API-ключів.
    """

    BASE_URL = "https://api.bybit.com/v5/market/kline"
    GRAM_REBRAND_DATE = datetime(2026, 6, 22, 0, 0, 0, tzinfo=UTC)

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    async def fetch_synthetic_kline(
            self,
            symbol: str = "GRAMUSDT",
            category: str = "linear",
            interval: str = "1",
            start_time: datetime | None = None,
            end_time: datetime | None = None,
    ) -> pl.DataFrame:
        """
        Завантажує та зшиває безперервну історію для TON/GRAM.
        Якщо symbol - GRAMUSDT або TONUSDT і діапазон перетинає червень 2026:
        1. Тягне TONUSDT до 22.06.2026
        2. Тягне GRAMUSDT після 22.06.2026
        3. Зшиває їх 1:1 по timestamp.
        Для інших тикерів (BTCUSDT) викликає звичайний fetch_kline.
        """
        clean_symbol = symbol.upper().strip()

        # Якщо запит не стосується TON/GRAM, робимо звичайний виклик
        if clean_symbol not in ("TONUSDT", "GRAMUSDT"):
            return await self.fetch_kline(
                symbol=clean_symbol,
                category=category,
                interval=interval,
                start_time=start_time,
                end_time=end_time,
            )

        dfs = []

        # 1. Запит для TONUSDT (до ребрендингу)
        if start_time is None or start_time < self.GRAM_REBRAND_DATE:
            ton_end = min(end_time, self.GRAM_REBRAND_DATE) if end_time else self.GRAM_REBRAND_DATE
            df_ton = await self.fetch_kline(
                symbol="TONUSDT",
                category=category,
                interval=interval,
                start_time=start_time,
                end_time=ton_end,
            )
            if not df_ton.is_empty():
                dfs.append(df_ton)

        # 2. Запит для GRAMUSDT (після ребрендингу)
        if end_time is None or end_time >= self.GRAM_REBRAND_DATE:
            gram_start = (
                max(start_time, self.GRAM_REBRAND_DATE) if start_time else self.GRAM_REBRAND_DATE
            )
            df_gram = await self.fetch_kline(
                symbol="GRAMUSDT",
                category=category,
                interval=interval,
                start_time=gram_start,
                end_time=end_time,
            )
            if not df_gram.is_empty():
                dfs.append(df_gram)

        if not dfs:
            return pl.DataFrame()

        # 3. Зшиваємо та сортуємо
        full_df = pl.concat(dfs).unique(subset=["timestamp"]).sort("datetime")
        return full_df

    async def fetch_kline(
            self,
            symbol: str = "BTCUSDT",
            category: str = "linear",
            interval: str = "1",
            start_time: datetime | None = None,
            end_time: datetime | None = None,
            limit: int = 1000,
    ) -> pl.DataFrame:
        """
        Завантажує свічки та повертає їх у Polars DataFrame.
        symbol: "BTCUSDT", "TONUSDT" тощо.
        category: "spot" або "linear" (ф'ючерси).
        interval: "1" (1m), "5" (5m), "60" (1h), "D" (1d).
        """
        clean_symbol = symbol.upper().strip()

        params_base: dict[str, str | int] = {
            "category": category,
            "symbol": clean_symbol,
            "interval": interval,
            "limit": limit,
        }

        start_ms = int(start_time.astimezone(UTC).timestamp() * 1000) if start_time else None
        end_ms = int(end_time.astimezone(UTC).timestamp() * 1000) if end_time else None

        all_records: list[list[Any]] = []

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(
                "Fetching %s (%s) klines from Bybit API from %s to %s...",
                clean_symbol,
                category,
                start_time,
                end_time,
            )

            # Single request when no time bounds provided
            if start_ms is None and end_ms is None:
                params = {**params_base}
                response = await client.get(self.BASE_URL, params=params)
                response.raise_for_status()
                data = response.json()
                if data.get("retCode") != 0:
                    err_msg = data.get("retMsg")
                    logger.error("Bybit API error code %s: %s", data.get("retCode"), err_msg)
                    raise ValueError(f"Bybit API Error ({clean_symbol}): {err_msg}")

                raw_list = data.get("result", {}).get("list", [])
                if not raw_list:
                    logger.warning("No kline data returned for %s", clean_symbol)
                    return pl.DataFrame()

                all_records.extend(raw_list)
            else:
                # Backwards pagination: request `limit` candles ending at
                # `current_end` and move backwards
                current_end = (
                    end_ms if end_ms is not None else int(datetime.now(UTC).timestamp() * 1000)
                )
                page = 0
                while True:
                    page += 1
                    params = {**params_base, "end": current_end}

                    response = await client.get(self.BASE_URL, params=params)
                    response.raise_for_status()
                    data = response.json()

                    if data.get("retCode") != 0:
                        err_msg = data.get("retMsg")
                        logger.error("Bybit API error code %s: %s", data.get("retCode"), err_msg)
                        raise ValueError(f"Bybit API Error ({clean_symbol}): {err_msg}")

                    raw_list = data.get("result", {}).get("list", [])
                    if not raw_list:
                        logger.info(
                            "No more kline data returned on page %s for %s",
                            page,
                            clean_symbol,
                        )
                        break

                    all_records.extend(raw_list)

                    # determine oldest timestamp in this batch
                    try:
                        ts_values = [int(item[0]) for item in raw_list]
                    except Exception:
                        logger.warning(
                            "Unexpected timestamp format in Bybit response page %s",
                            page,
                        )
                        break

                    min_ts = min(ts_values)

                    # stop if we've reached the requested start boundary
                    if start_ms is not None and min_ts <= start_ms:
                        logger.info("Reached start_time on page %s for %s", page, clean_symbol)
                        break

                    # If the API returned fewer than requested, assume no more data
                    if len(raw_list) < limit:
                        logger.info(
                            "Received final page %s (len=%s) for %s",
                            page,
                            len(raw_list),
                            clean_symbol,
                        )
                        break

                    # Move the end pointer to just before the oldest timestamp to page backwards
                    current_end = min_ts - 1

                    await asyncio.sleep(0.05)

        # No records collected
        if not all_records:
            logger.warning("No kline data collected for %s", clean_symbol)
            return pl.DataFrame()

        # Build unique records keyed by timestamp to avoid duplicates
        unique_map: dict[int, list[Any]] = {}
        for item in all_records:
            try:
                ts = int(item[0])
            except Exception:
                continue
            unique_map[ts] = item

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
            for ts, item in sorted(unique_map.items())
        ]

        df = pl.DataFrame(records)
        df = df.with_columns(
            pl.from_epoch("timestamp", time_unit="ms").dt.replace_time_zone("UTC").alias("datetime")
        ).sort("datetime")

        logger.success("Successfully fetched %s candles for %s", len(df), clean_symbol)
        return df