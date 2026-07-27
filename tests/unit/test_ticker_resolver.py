from datetime import datetime, timedelta
import pytest

from core.ticker_resolver import TickerResolver, GRAM_REBRAND_DATE

class TestTickerResolver:

    @pytest.mark.parametrize(
        "dt,expected",
        [
            (datetime(2023, 5, 15, 12, 0), "TONUSDT"),
            (datetime(2026, 6, 15, 23, 59), "TONUSDT"),
            (GRAM_REBRAND_DATE - timedelta(seconds=1), "TONUSDT"),
            (GRAM_REBRAND_DATE, "GRAMUSDT"),
            (datetime(2026, 6, 23, 0, 0), "GRAMUSDT"),
        ],
    )
    def test_resolve_ticker_boundaries(self, dt: datetime, expected: str):
        """ Перевірка визначення тикера на межах дати ребрендингу """
        assert TickerResolver.resolve_ticker(dt) == expected

    def test_resolve_ticker_invalid_type(self):
        """ Перевірка негативного кейсу з некоректним типом """
        with pytest.raises(TypeError):
            TickerResolver.resolve_ticker(None)  # type: ignore

    def test_get_required_tickers_historical_only(self):
        """ Тільки період TONUSDT """
        start = datetime(2023, 1, 1)
        end = datetime(2026, 1, 1)
        tickers = TickerResolver.get_required_tickers(start, end)
        assert tickers == ["TONUSDT", "BTCUSDT"]

    def test_get_required_tickers_spanning_both_eras(self):
        """ Період, що перетинає дату ребрендингу """
        start = datetime(2026, 1, 1)
        end = datetime(2026, 7, 1)
        tickers = TickerResolver.get_required_tickers(start, end)
        assert tickers == ["TONUSDT", "GRAMUSDT", "BTCUSDT"]

    def test_get_required_tickers_invalid_range(self):
        """ Валідація некоректного часового діапазону (start > end) """
        start = datetime(2026, 7, 1)
        end = datetime(2026, 1, 1)
        with pytest.raises(ValueError, match="start_date cannot be after end_date"):
            TickerResolver.get_required_tickers(start, end)
