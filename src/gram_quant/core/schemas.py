from datetime import datetime

from pydantic import BaseModel, Field


class NewsEventRaw(BaseModel):
    """
    Модель сирої події з Excel-таблиці.
    Відповідає структурі ручних спостережень за постами Павла Дурова.
    """
    event_time_kyiv: datetime = Field(description="Час події (За київським часом)")
    author: str = Field(default="Pavel Durov", description="Автор публікації")
    url: str | None = Field(default=None, description="Посилання на пост")
    text: str = Field(description="Текст повідомлення")
    importance: int = Field(ge=1, le=5, description="Ручна оцінка важливості (1-5)")

    # Початкові ринкові показники
    price_before: float = Field(gt=0, description="Ціна GRAM до повідомлення")
    btc_before: float = Field(gt=0, description="Ціна BTC до повідомлення")
    volume_before: float | None = Field(default=None, ge=0, description="Об'єм торгів до події")

    # Історичні зрізи ціни після події
    price_5m: float | None = Field(default=None, gt=0)
    price_15m: float | None = Field(default=None, gt=0)
    price_1h: float | None = Field(default=None, gt=0)
    price_6h: float | None = Field(default=None, gt=0)
    price_24h: float | None = Field(default=None, gt=0)

    # Екстремуми та розрахований Return з таблиці
    max_price: float | None = Field(default=None, gt=0)
    min_price: float | None = Field(default=None, gt=0)
    peak_time: datetime | None = Field(default=None, description="Час пікового значення ціни")
    manual_return: float | None = Field(default=None, description="Ручний return із таблиці")


class EventWindowRequest(BaseModel):
    """Запит на вирізку часового вікна для кількісного аналізу."""
    event_id: str
    event_time_utc: datetime
    pre_window_minutes: int = 60
    post_window_minutes: int = 1440  # 24 години
