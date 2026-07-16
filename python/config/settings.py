"""
应用配置 — 通过环境变量或 .env 文件加载
"""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def normalize_openai_base_url(base_url: str) -> str:
    """Accept either an OpenAI-compatible base URL or a concrete chat endpoint."""
    value = base_url.strip().rstrip("/")
    if not value:
        return "https://api.openai.com/v1"
    for suffix in ("/chat/completions", "/completions"):
        if value.endswith(suffix):
            return value[: -len(suffix)]
    return value


class Settings(BaseSettings):
    # LLM
    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    qa_context_limit: int = Field(default=8, ge=1, le=50)
    deepresearch_max_iterations: int = Field(default=3, ge=1, le=10)
    deepresearch_queries_per_iteration: int = Field(default=3, ge=1, le=10)
    deepresearch_contexts_per_query: int = Field(default=4, ge=1, le=20)
    deepresearch_final_context_limit: int = Field(default=12, ge=1, le=100)
    deepresearch_min_evidence: int = Field(default=2, ge=1, le=20)
    deepresearch_min_sources: int = Field(default=2, ge=1, le=10)
    deepsearch_max_iterations: int = Field(default=3, ge=1, le=10)
    deepsearch_queries_per_iteration: int = Field(default=3, ge=1, le=10)
    deepsearch_contexts_per_query: int = Field(default=4, ge=1, le=20)
    deepsearch_final_context_limit: int = Field(default=12, ge=1, le=100)
    web_search_provider: Literal["disabled", "tavily", "serpapi", "duckduckgo"] = "duckduckgo"
    tavily_api_key: str = ""
    serpapi_api_key: str = ""
    web_search_timeout_seconds: int = Field(default=10, ge=1, le=60)
    web_search_top_k: int = Field(default=5, ge=1, le=20)
    web_page_fetch_enabled: bool = True
    web_page_timeout_seconds: int = Field(default=10, ge=1, le=60)
    web_page_max_chars: int = Field(default=12000, ge=1000, le=200000)
    web_page_chunks_per_result: int = Field(default=2, ge=1, le=20)
    web_page_block_private_networks: bool = True
    run_trace_dir: str = "./run_traces"
    dependency_init_timeout_seconds: int = Field(default=3, ge=1, le=60)

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Vector Store
    vector_store_type: Literal["chroma", "pgvector"] = "chroma"
    chroma_host: str = "localhost"
    chroma_port: int = Field(default=8000, ge=1, le=65535)
    pgvector_dsn: str = "postgresql://postgres:postgres@localhost:5432/knowledge"

    # Kafka (CDC)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_doc_changes: str = "doc-changes"
    kafka_topic_kg_updates: str = "kg-updates"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8080, ge=1, le=65535)

    # Document Store
    upload_dir: str = "./uploads"
    upload_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1024, le=1024 * 1024 * 1024)

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def openai_client_base_url(self) -> str:
        return normalize_openai_base_url(self.openai_base_url)


settings = Settings()
