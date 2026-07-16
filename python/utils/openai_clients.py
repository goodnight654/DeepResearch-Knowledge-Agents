"""Factories for OpenAI-compatible LangChain clients.

Client construction should not make the API process unimportable when a key is
not configured. The placeholder only defers the actionable error until an LLM
or embedding call is actually attempted; compatible local gateways can still
override it through the normal environment variables.
"""

from __future__ import annotations

from typing import Any

from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from config import settings


class UnavailableChatModel:
    async def ainvoke(self, messages: Any) -> Any:
        raise RuntimeError("OPENAI_API_KEY is not configured")


class UnavailableEmbeddings:
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("OPENAI_API_KEY is not configured")

    async def aembed_query(self, text: str) -> list[float]:
        raise RuntimeError("OPENAI_API_KEY is not configured")


def create_chat_model(*, temperature: float = 0, **kwargs: Any) -> Any:
    api_key = settings.openai_api_key.strip()
    if not api_key:
        return UnavailableChatModel()
    return ChatOpenAI(
        model=settings.openai_model,
        api_key=api_key,
        base_url=settings.openai_client_base_url,
        temperature=temperature,
        **kwargs,
    )


def create_embeddings(**kwargs: Any) -> Any:
    api_key = settings.openai_api_key.strip()
    if not api_key:
        return UnavailableEmbeddings()
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=api_key,
        base_url=settings.openai_client_base_url,
        **kwargs,
    )
