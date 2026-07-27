from datetime import datetime
from typing import Literal

# Точна межа ребрендингу: 22 червня 2026 року 00:00:00
GRAM_REBRAND_DATE = datetime(2026, 6, 22, 0, 0, 0)

TickerType = Literal["TONUSDT", "GRAMUSDT"]


class TickerResolver:
    """
    Квант-модуль для мапінгу часових міток подій на відповідні тикери
    з урахуванням ребрендингу TON -> GRAM у червні 2026 року.
    """

    @staticmethod
    def resolve_ticker(event_timestamp: datetime) -> TickerType:
        """
        Визначає цільовий тикер залежно від дати та часу події.
        """
        if not isinstance(event_timestamp, datetime):
            raise TypeError(f"Expected datetime object, got {type(event_timestamp).__name__}")

        if event_timestamp < GRAM_REBRAND_DATE:
            return "TONUSDT"
        return "GRAMUSDT"

    @staticmethod
    def get_required_tickers(start_date: datetime, end_date: datetime) -> list[str]:
        """
        Повертає впорядкований список необхідних тикерів для завантаження
        історії у заданому діапазоні дат + обов'язковий BTCUSDT бенчмарк.
        """
        if not isinstance(start_date, datetime) or not isinstance(end_date, datetime):
            raise TypeError("start_date and end_date must be datetime instances")

        if start_date > end_date:
            raise ValueError("start_date cannot be after end_date")

        tickers: list[str] = []

        # Якщо діапазон чіпає до-ребрендингову еру
        if start_date < GRAM_REBRAND_DATE:
            tickers.append("TONUSDT")

        # Якщо діапазон чіпає після-ребрендингову еру
        if end_date >= GRAM_REBRAND_DATE:
            tickers.append("GRAMUSDT")

        # Обов'язково додаємо ринковий бенчмарк для очищення бети
        tickers.append("BTCUSDT")

        return tickers
