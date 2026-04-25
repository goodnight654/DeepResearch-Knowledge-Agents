import sys
import types
import asyncio


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


class _OpenAIEmbeddings:
    def __init__(self, *args, **kwargs):
        pass


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

from services.hybrid_retriever import HybridRetriever, QueryAnalysis
from services.web_page_reader import WebPageChunk, WebPageReader
from services.web_search import WebSearchResult


class FakeWebSearch:
    enabled = True

    def __init__(self):
        self.calls = []

    async def search(self, query, top_k=None):
        self.calls.append((query, top_k))
        return [
            WebSearchResult(
                title="Result title",
                url="https://example.com/article",
                snippet="Snippet only",
                score=0.9,
                provider="fake",
            )
        ]


class FakePageReader:
    enabled = True

    def __init__(self):
        self.calls = []

    async def read(self, url, title="", score=1.0, max_chunks=None):
        self.calls.append(
            {
                "url": url,
                "title": title,
                "score": score,
                "max_chunks": max_chunks,
            }
        )
        return [
            WebPageChunk(
                url=url,
                title=title,
                content="Full article body with enough detail for evidence.",
                chunk_index=0,
                retrieved_at="2026-04-24T00:00:00+00:00",
                metadata={"score": score},
            )
        ]


def test_web_page_reader_extracts_visible_text():
    title, text = WebPageReader._extract_text(
        """
        <html>
          <head><title>Research Page</title><style>.hidden{}</style></head>
          <body>
            <script>ignoreMe()</script>
            <h1>Visible heading</h1>
            <p>Important paragraph.</p>
          </body>
        </html>
        """
    )

    assert title == "Research Page"
    assert "Visible heading" in text
    assert "Important paragraph." in text
    assert "ignoreMe" not in text


def test_hybrid_retriever_prefers_page_chunks_over_snippets():
    asyncio.run(_assert_hybrid_retriever_prefers_page_chunks_over_snippets())


async def _assert_hybrid_retriever_prefers_page_chunks_over_snippets():
    web_search = FakeWebSearch()
    page_reader = FakePageReader()
    retriever = HybridRetriever(
        web_search=web_search,
        web_page_reader=page_reader,
        llm=object(),
    )

    contexts = await retriever._web_retrieve(
        QueryAnalysis(question="q", queries=["q"]),
        top_k=2,
    )

    assert len(contexts) == 1
    assert contexts[0].retrieval_type == "web_page"
    assert contexts[0].content.startswith("Full article body")
    assert contexts[0].metadata["url"] == "https://example.com/article"
    assert web_search.calls == [("q", 2)]
    assert page_reader.calls[0]["url"] == "https://example.com/article"
