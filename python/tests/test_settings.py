import sys
import types


pydantic_settings = types.ModuleType("pydantic_settings")


class _BaseSettings:
    def __init__(self, **kwargs):
        for name, value in self.__class__.__dict__.items():
            if name.startswith("_") or callable(value) or isinstance(value, property):
                continue
            setattr(self, name, kwargs.get(name, value))


pydantic_settings.BaseSettings = _BaseSettings
sys.modules.setdefault("pydantic_settings", pydantic_settings)

from config.settings import Settings, normalize_openai_base_url


def test_normalize_openai_base_url_accepts_chat_endpoint():
    assert (
        normalize_openai_base_url("https://models.sjtu.edu.cn/api/v1/chat/completions")
        == "https://models.sjtu.edu.cn/api/v1"
    )


def test_settings_exposes_client_base_url():
    settings = Settings(openai_base_url="https://models.sjtu.edu.cn/api/v1/chat/completions/")

    assert settings.openai_client_base_url == "https://models.sjtu.edu.cn/api/v1"
