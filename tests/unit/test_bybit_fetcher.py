from datetime import UTC, datetime
from unittest.mock import patch

import polars as pl
import pytest

from gram_quant.fetchers.bybit import BybitFetcher


class TestBybitFetcherSynthetic:
    @pytest.mark.asyncio
    @patch.object(BybitFetcher, "fetch_kline")
    async def test_fetch_synthetic_kline_spans_rebrand(self, mock_fetch_kline):
        """Перевіряє, що synthetic метод робить 2 запити для TONUSDT та GRAMUSDT"""
        df_ton = pl.DataFrame(
            {
                "timestamp": [1700000000000],
                "open": [1.0],
                "high": [1.1],
                "low": [0.9],
                "close": [1.05],
                "volume": [100.0],
                "turnover": [105.0],
                "datetime": [datetime(2026, 6, 1, tzinfo=UTC)],
            }
        )
        df_gram = pl.DataFrame(
            {
                "timestamp": [1719000000000],
                "open": [1.05],
                "high": [1.2],
                "low": [1.0],
                "close": [1.15],
                "volume": [200.0],
                "turnover": [230.0],
                "datetime": [datetime(2026, 6, 23, tzinfo=UTC)],
            }
        )

        mock_fetch_kline.side_effect = [df_ton, df_gram]

        fetcher = BybitFetcher()
        start = datetime(2026, 6, 1, tzinfo=UTC)
        end = datetime(2026, 6, 30, tzinfo=UTC)

        result = await fetcher.fetch_synthetic_kline(
            symbol="GRAMUSDT", start_time=start, end_time=end
        )

        assert len(result) == 2
        assert mock_fetch_kline.call_count == 2
