from agents.deepresearch_agent import DeepResearchAgent, DeepResearchEvidence
from services.hybrid_retriever import RetrievedContext


class FailingLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("offline")


def _agent() -> DeepResearchAgent:
    return DeepResearchAgent(llm=FailingLLM(), retriever=object())


def test_merge_contexts_prefers_high_score_and_deduplicates():
    agent = _agent()
    contexts = [
        RetrievedContext("same content", "source", 0.6, "vector"),
        RetrievedContext("same content", "source", 0.9, "vector"),
        RetrievedContext("graph content", "knowledge_graph", 0.5, "graph"),
    ]

    merged = agent.merge_contexts(contexts)

    assert len(merged) == 2
    assert merged[0].content == "same content"
    assert merged[0].score == 0.9


def test_clean_list_and_select_queries_remove_repeats():
    agent = _agent()
    assert agent.clean_list(["query a", "", "query a", " query b ", None]) == ["query a", "query b"]
    assert agent.select_queries(["query a", "query b"], {"query a"}) == ["query b"]


def test_select_final_contexts_prefers_independent_sources():
    agent = _agent()
    agent.final_context_limit = 2
    contexts = [
        RetrievedContext("a1", "https://a.example", 0.99, "web_page"),
        RetrievedContext("a2", "https://a.example", 0.98, "web_page"),
        RetrievedContext("b1", "https://b.example", 0.70, "web_page"),
    ]
    assert [item.content for item in agent.select_final_contexts(contexts)] == ["a1", "b1"]


def test_build_evidence_assigns_stable_ids_and_clamps_confidence():
    evidence = DeepResearchAgent.build_evidence(
        [
            RetrievedContext(
                content="Detailed evidence from a fetched page.",
                source="https://example.com/research",
                score=1.4,
                retrieval_type="web_page",
                metadata={"title": "Research Article", "retrieved_at": "2026-04-24T00:00:00+00:00"},
            )
        ]
    )
    assert evidence[0].evidence_id == "E1"
    assert evidence[0].confidence == 1.0
    assert evidence[0].title == "Research Article"


def test_validate_citations_removes_unknown_ids():
    evidence = [DeepResearchEvidence("E1", "source", "Title", "quote", 0.8, "2026-01-01T00:00:00+00:00")]
    answer, cited, warnings = DeepResearchAgent.validate_citations("Claim [E1], bad [E9].", evidence)
    assert "[E1]" in answer and "[E9]" not in answer
    assert cited == ["E1"]
    assert warnings == ["已移除不存在的证据引用: E9"]


def test_validate_citations_adds_index_and_no_evidence_forces_zero_confidence():
    evidence = [DeepResearchEvidence("E1", "source", "Title", "quote", 0.8, "2026-01-01T00:00:00+00:00")]
    answer, cited, warnings = DeepResearchAgent.validate_citations("Uncited claim.", evidence)
    assert "证据索引" in answer
    assert cited == ["E1"]
    assert warnings
    assert DeepResearchAgent.calibrate_confidence(0.99, [], []) == 0.0


async def test_offline_research_stops_when_two_independent_sources_are_available():
    class FakeRetriever:
        async def retrieve(self, question, **kwargs):
            return None, [
                RetrievedContext("first evidence", "https://a.example", 0.9, "web_page"),
                RetrievedContext("second evidence", "https://b.example", 0.8, "web_page"),
            ]

    agent = DeepResearchAgent(llm=FailingLLM(), retriever=FakeRetriever())
    result = await agent.research("How does the system work?")

    assert result.iterations == 1
    assert result.cited_evidence_ids == ["E1", "E2"]
    assert "[E1]" in result.answer and "[E2]" in result.answer
    assert result.confidence > 0
