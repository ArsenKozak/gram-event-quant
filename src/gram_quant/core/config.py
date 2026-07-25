from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "GramEventQuant"
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"

    telegram_api_id: int | None = None
    telegram_api_hash: SecretStr | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
