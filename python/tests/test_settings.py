import pytest
from pydantic import ValidationError

from config.settings import Settings, normalize_openai_base_url


def test_normalize_openai_base_url_accepts_chat_endpoint():
    assert (
        normalize_openai_base_url("https://models.example.com/api/v1/chat/completions/")
        == "https://models.example.com/api/v1"
    )


def test_normalize_openai_base_url_uses_default_for_blank_value():
    assert normalize_openai_base_url("  ") == "https://api.openai.com/v1"


def test_settings_exposes_client_base_url():
    app_settings = Settings(openai_base_url="https://models.example.com/api/v1/chat/completions/")
    assert app_settings.openai_client_base_url == "https://models.example.com/api/v1"


def test_settings_rejects_invalid_research_limits():
    with pytest.raises(ValidationError):
        Settings(deepresearch_max_iterations=0)


def test_settings_rejects_unknown_web_provider():
    with pytest.raises(ValidationError):
        Settings(web_search_provider="unknown")


def test_settings_rejects_unknown_vector_backend():
    with pytest.raises(ValidationError):
        Settings(vector_store_type="unknown")
