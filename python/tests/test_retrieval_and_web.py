from services.hybrid_retriever import HybridRetriever, QueryAnalysis, RetrievedContext
from services.web_page_reader import WebPageChunk, WebPageReader
from services.web_search import WebSearchResult, WebSearchService, _DuckDuckGoHTMLParser


class FailingLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("offline")


class FailingVectorStore:
    async def search(self, query, top_k):
        raise ConnectionError("vector service is down")


class FakeWebSearch:
    enabled = True

    def __init__(self):
        self.calls = []

    async def search(self, query, top_k=None):
        self.calls.append((query, top_k))
        return [WebSearchResult("Result title", "https://example.com/article", "Snippet", 0.9, "fake")]


class FakePageReader:
    enabled = True

    async def read(self, url, title="", score=1.0, max_chunks=None):
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


def test_web_page_reader_extracts_visible_text_and_skips_navigation():
    title, text = WebPageReader._extract_text(
        """
        <html><head><title>Research Page</title><style>.hidden{}</style></head>
        <body><script>ignoreMe()</script><nav>menu item</nav>
        <h1>Visible heading</h1><p>Important paragraph.</p></body></html>
        """
    )
    assert title == "Research Page"
    assert "Visible heading" in text and "Important paragraph." in text
    assert "ignoreMe" not in text and "menu item" not in text


def test_web_page_reader_blocks_private_targets():
    reader = WebPageReader()
    assert not reader._is_allowed_url("http://127.0.0.1/admin", resolve_dns=False)
    assert not reader._is_allowed_url("http://localhost/admin", resolve_dns=False)
    assert reader._is_allowed_url("https://example.com/page", resolve_dns=False)


def test_web_page_chunking_uses_overlap_and_chinese_boundaries():
    text = ("第一段内容。" * 180) + ("第二段内容。" * 180)
    chunks = WebPageReader._chunk_text(text, max_chunks=3, chunk_size=500, overlap=50)
    assert len(chunks) == 3
    assert len(set(chunks)) == 3
    assert any(chunks[0][-30:] in chunk for chunk in chunks[1:])


async def test_hybrid_retriever_prefers_page_chunks_and_survives_vector_failure():
    web_search = FakeWebSearch()
    retriever = HybridRetriever(
        vector_store=FailingVectorStore(),
        web_search=web_search,
        web_page_reader=FakePageReader(),
        llm=FailingLLM(),
    )

    analysis, contexts = await retriever.retrieve("research question", top_k=2)

    assert analysis.queries == ["research question"]
    assert len(contexts) == 1
    assert contexts[0].retrieval_type == "web_page"
    assert contexts[0].content.startswith("Full article body")
    assert web_search.calls == [("research question", 2)]


def test_hybrid_rerank_does_not_mutate_input_and_clamps_score():
    original = RetrievedContext(" body  with   spaces ", "source", 0.95, "graph")
    ranked = HybridRetriever._hybrid_rerank([original])
    assert original.score == 0.95
    assert ranked[0].score == 1.0
    assert ranked[0].content == "body with spaces"


def test_model_generated_cypher_must_be_read_only():
    assert HybridRetriever._is_read_only_cypher("MATCH (n) RETURN n LIMIT 5")
    assert not HybridRetriever._is_read_only_cypher("MATCH (n) DELETE n")
    assert not HybridRetriever._is_read_only_cypher("CALL db.labels()")


def test_duckduckgo_html_parser_and_redirect_unwrap():
    parser = _DuckDuckGoHTMLParser()
    parser.feed(
        '<div class="result"><a class="result__a" href="https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fx">'
        'Example result</a><a class="result__snippet">Useful snippet</a></div>'
    )
    assert parser.results[0]["title"] == "Example result"
    assert parser.results[0]["snippet"] == "Useful snippet"
    assert WebSearchService._unwrap_duck_url(parser.results[0]["url"]) == "https://example.com/x"


async def test_web_retrieve_falls_back_to_snippet_when_page_read_fails():
    class EmptyReader:
        enabled = True

        async def read(self, **kwargs):
            return []

    retriever = HybridRetriever(web_search=FakeWebSearch(), web_page_reader=EmptyReader(), llm=FailingLLM())
    contexts = await retriever._web_retrieve(QueryAnalysis("q", ["q"]), top_k=1)
    assert contexts[0].retrieval_type == "web"
    assert contexts[0].source == "https://example.com/article"
