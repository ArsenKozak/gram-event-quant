from gram_quant.core.config import Settings


def test_settings_load():
    settings = Settings()
    assert settings.app_name == "GramEventQuant"