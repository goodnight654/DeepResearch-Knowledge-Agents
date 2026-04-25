import sys
import types


langchain_core = types.ModuleType("langchain_core")
langchain_core_messages = types.ModuleType("langchain_core.messages")
langchain_openai = types.ModuleType("langchain_openai")
pydantic_settings = types.ModuleType("pydantic_settings")


class _Message:
    def __init__(self, content):
        self.content = content


class _ChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass

    async def ainvoke(self, messages):
        raise RuntimeError("stub llm should not be used")


class _OpenAIEmbeddings:
    def __init__(self, *args, **kwargs):
        pass

    async def aembed_documents(self, texts):
        return [[0.0] for _ in texts]

    async def aembed_query(self, query):
        return [0.0]


class _BaseSettings:
    def __init__(self, **kwargs):
        for name, value in self.__class__.__dict__.items():
            if name.startswith("_") or callable(value) or isinstance(value, property):
                continue
            setattr(self, name, kwargs.get(name, value))


langchain_core_messages.HumanMessage = _Message
langchain_core_messages.SystemMessage = _Message
langchain_openai.ChatOpenAI = _ChatOpenAI
langchain_openai.OpenAIEmbeddings = _OpenAIEmbeddings
pydantic_settings.BaseSettings = _BaseSettings

sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.messages", langchain_core_messages)
sys.modules.setdefault("langchain_openai", langchain_openai)
sys.modules.setdefault("pydantic_settings", pydantic_settings)

from agents.deepsearch_agent import DeepSearchAgent
from services.hybrid_retriever import RetrievedContext


class FakeLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("llm should not be called in this unit test")


def test_hybrid_rerank_prefers_high_score_and_dedupes():
    agent = DeepSearchAgent(llm=FakeLLM(), retriever=object())
    contexts = [
        RetrievedContext(
            content="same content",
            source="vector_store",
            score=0.6,
            retrieval_type="vector",
        ),
        RetrievedContext(
            content="same content",
            source="vector_store",
            score=0.9,
            retrieval_type="vector",
        ),
        RetrievedContext(
            content="graph content",
            source="knowledge_graph",
            score=0.5,
            retrieval_type="graph",
        ),
    ]

    reranked = agent._merge_contexts(contexts)

    assert len(reranked) == 2
    assert reranked[0].content == "same content"
    assert reranked[0].score == 0.9


def test_clean_list_removes_empty_and_duplicates():
    values = ["query a", "", "query a", " query b ", None]

    cleaned = DeepSearchAgent._clean_list(values)

    assert cleaned == ["query a", "query b"]


def test_build_evidence_assigns_stable_ids_and_sources():
    contexts = [
        RetrievedContext(
            content="Detailed evidence from a fetched page.",
            source="https://example.com/research",
            score=0.95,
            retrieval_type="web_page",
            metadata={
                "title": "Research Article",
                "retrieved_at": "2026-04-24T00:00:00+00:00",
            },
        )
    ]

    evidence = DeepSearchAgent.build_evidence(contexts)

    assert evidence[0].evidence_id == "E1"
    assert evidence[0].source == "https://example.com/research"
    assert evidence[0].title == "Research Article"
    assert evidence[0].quote == "Detailed evidence from a fetched page."
