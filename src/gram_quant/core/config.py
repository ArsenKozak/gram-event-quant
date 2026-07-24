from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "GramEventQuant"
    data_raw_dir: str = "data/raw"
    data_processed_dir: str = "data/processed"


settings = Settings()