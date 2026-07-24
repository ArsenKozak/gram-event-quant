from unittest.mock import patch

import pytest

from gram_quant.fetchers.bybit import BybitFetcher


@pytest.mark.asyncio
async def test_bybit_fetch_kline_mock():
    """Тестування парсингу відповіді Bybit API через Mock."""
    mock_response = {
        "retCode": 0,
        "retMsg": "OK",
        "result": {
            "category": "spot",
            "symbol": "BTCUSDT",
            "list": [
                ["1700000000000", "65000.0", "65500.0", "64900.0", "65400.0", "120.5", "7850000"],
                ["1700000060000", "65400.0", "65800.0", "65350.0", "65750.0", "180.2", "11840000"],
            ],
        },
    }

    with patch("httpx.AsyncClient.get") as mock_get:
        mock_get.return_value.status_code = 200
        mock_get.return_value.json = lambda: mock_response
        mock_get.return_value.raise_for_status = lambda: None

        fetcher = BybitFetcher()
        df = await fetcher.fetch_kline(symbol="BTCUSDT", interval="1")

        assert len(df) == 2
        assert "datetime" in df.columns
        assert df["close"][0] == 65400.0
        assert df["volume"][1] == 180.2


@pytest.mark.asyncio
async def test_bybit_live_fetch_btc():
    """Справжній мережевий тест до публічного API Bybit (BTCUSDT)."""
    fetcher = BybitFetcher()
    df = await fetcher.fetch_kline(symbol="BTCUSDT", category="spot", limit=5)

    assert not df.is_empty()
    assert len(df) == 5
    assert "close" in df.columns
