"""
应用配置 — 通过环境变量或 .env 文件加载
"""

from pydantic_settings import BaseSettings


def normalize_openai_base_url(base_url: str) -> str:
    """Accept either an OpenAI-compatible base URL or a concrete chat endpoint."""
    value = base_url.strip().rstrip("/")
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
    qa_context_limit: int = 8
    deepresearch_max_iterations: int = 3
    deepresearch_queries_per_iteration: int = 3
    deepresearch_contexts_per_query: int = 4
    deepresearch_final_context_limit: int = 12
    deepsearch_max_iterations: int = 3
    deepsearch_queries_per_iteration: int = 3
    deepsearch_contexts_per_query: int = 4
    deepsearch_final_context_limit: int = 12
    web_search_provider: str = "duckduckgo"  # disabled | tavily | serpapi | duckduckgo
    tavily_api_key: str = ""
    serpapi_api_key: str = ""
    web_search_timeout_seconds: int = 10
    web_search_top_k: int = 5
    web_page_fetch_enabled: bool = True
    web_page_timeout_seconds: int = 10
    web_page_max_chars: int = 12000
    web_page_chunks_per_result: int = 2
    run_trace_dir: str = "./run_traces"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    # Vector Store
    vector_store_type: str = "chroma"  # chroma | pgvector
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    pgvector_dsn: str = "postgresql://postgres:postgres@localhost:5432/knowledge"

    # Kafka (CDC)
    kafka_bootstrap_servers: str = "localhost:9092"
    kafka_topic_doc_changes: str = "doc-changes"
    kafka_topic_kg_updates: str = "kg-updates"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080

    # Document Store
    upload_dir: str = "./uploads"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    @property
    def openai_client_base_url(self) -> str:
        return normalize_openai_base_url(self.openai_base_url)


settings = Settings()
