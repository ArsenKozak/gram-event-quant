from datetime import UTC, datetime

import httpx
import polars as pl
from loguru import logger


class BybitFetcher:
    """
    Асинхронний фетчер для завантаження публічних OHLCV свічок з Bybit V5 API.
    Працює без реєстрації та API-ключів.
    """

    BASE_URL = "https://api.bybit.com/v5/market/kline"

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    async def fetch_kline(
        self,
        symbol: str = "BTCUSDT",
        category: str = "spot",
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
        params: dict[str, str | int] = {
            "category": category,
            "symbol": clean_symbol,
            "interval": interval,
            "limit": limit,
        }

        if start_time:
            params["start"] = int(start_time.astimezone(UTC).timestamp() * 1000)
        if end_time:
            params["end"] = int(end_time.astimezone(UTC).timestamp() * 1000)

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            logger.info(f"Fetching {clean_symbol} ({category}) klines from Bybit API...")
            response = await client.get(self.BASE_URL, params=params)
            response.raise_for_status()
            data = response.json()

        if data.get("retCode") != 0:
            err_msg = data.get("retMsg")
            logger.error(f"Bybit API error code {data.get('retCode')}: {err_msg}")
            raise ValueError(f"Bybit API Error ({clean_symbol}): {err_msg}")

        raw_list = data.get("result", {}).get("list", [])
        if not raw_list:
            logger.warning(f"No kline data returned for {clean_symbol}")
            return pl.DataFrame()

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
            for item in raw_list
        ]

        df = pl.DataFrame(records)
        df = df.with_columns(
            pl.from_epoch("timestamp", time_unit="ms").dt.replace_time_zone("UTC").alias("datetime")
        ).sort("datetime")

        logger.success(f"Successfully fetched {len(df)} candles for {clean_symbol}")
        return df
