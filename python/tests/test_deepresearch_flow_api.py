import io
from pathlib import Path

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from pydantic import ValidationError

from agents.deepresearch_agent import DeepResearchEvidence, DeepResearchResult, DeepResearchStep
from orchestrator.graph import _build_deepresearch_graph
from services.hybrid_retriever import RetrievedContext
from services.run_trace_store import RunTraceStore


class FakeDeepResearchAgent:
    def __init__(self):
        self.max_iterations = 3
        self.queries_per_iteration = 1
        self.final_context_limit = 5
        self.retrieve_calls = []
        self.gap_calls = 0

    @staticmethod
    def normalize_question(question):
        return question.strip()

    async def plan_search(self, question):
        return {"goal": question, "initial_queries": ["initial query"], "expected_evidence": ["web pages"]}

    def initial_frontier(self, plan, question):
        return list(plan["initial_queries"])

    def select_queries(self, candidates, attempted, fallback=""):
        values = candidates or ([fallback] if fallback else [])
        return [value for value in values if value.casefold() not in attempted][:1]

    async def retrieve_round(self, queries):
        self.retrieve_calls.append(list(queries))
        return [
            RetrievedContext(
                content=f"Evidence collected for {queries[0]}",
                source=f"https://example.com/{len(self.retrieve_calls)}",
                score=0.8,
                retrieval_type="web_page",
                metadata={"title": f"Page {len(self.retrieve_calls)}"},
            )
        ]

    def merge_contexts(self, contexts):
        return contexts

    def select_final_contexts(self, contexts):
        return contexts[: self.final_context_limit]

    async def summarize_round(self, question, focus, current_queries, all_contexts):
        return f"summary for {current_queries[0]}"

    async def analyze_gaps(self, question, focus, steps, summary, contexts):
        self.gap_calls += 1
        if self.gap_calls == 1:
            return {
                "answered": False,
                "confidence": 0.45,
                "next_focus": "missing detail",
                "gaps": ["need second source"],
                "follow_up_queries": ["follow up query"],
            }
        return {
            "answered": True,
            "confidence": 0.9,
            "next_focus": "done",
            "gaps": [],
            "follow_up_queries": [],
        }

    @staticmethod
    def clean_list(raw):
        return [str(item).strip() for item in raw or [] if str(item).strip()]

    def should_finalize(self, iteration, gap_state, follow_up_queries):
        return bool(gap_state.get("answered")) or iteration >= self.max_iterations or not follow_up_queries

    def make_step(self, iteration, focus, queries, summary, gaps, contexts):
        return DeepResearchStep(iteration, focus, queries, summary, gaps, contexts)

    @staticmethod
    def build_evidence(contexts):
        return [
            DeepResearchEvidence(
                f"E{index}",
                context.source,
                context.metadata["title"],
                context.content,
                context.score,
                "2026-04-24T00:00:00+00:00",
            )
            for index, context in enumerate(contexts, start=1)
        ]

    async def compose_answer(self, **kwargs):
        return "summary", "answer with citation [E1]", 0.82, ["E1"], []

    @staticmethod
    def build_result(**kwargs):
        return DeepResearchResult(**kwargs)


async def test_deepresearch_graph_runs_multiple_rounds_and_synthesizes_evidence():
    fake_agent = FakeDeepResearchAgent()
    graph = _build_deepresearch_graph(fake_agent)

    state = await graph.ainvoke({"question": "How should we build DeepResearch?"})

    assert fake_agent.retrieve_calls == [["initial query"], ["follow up query"]]
    assert state["result"].iterations == 2
    assert state["result"].cited_evidence_ids == ["E1"]
    assert [item["event"] for item in state["trace"]] == [
        "plan",
        "retrieve_round",
        "summarize_round",
        "assess_gap",
        "retrieve_round",
        "summarize_round",
        "assess_gap",
        "synthesize",
    ]


class FakeWorkflow:
    async def ainvoke(self, state):
        context = RetrievedContext(
            "Fetched article body",
            "https://example.com/research",
            0.92,
            "web_page",
            {"title": "Research Page"},
        )
        evidence = DeepResearchEvidence(
            "E1", context.source, "Research Page", context.content, 0.92, "2026-04-24T00:00:00+00:00"
        )
        step = DeepResearchStep(1, "research focus", ["deepresearch query"], "round summary", [], [context])
        return {
            "result": DeepResearchResult(
                state["question"],
                "final answer [E1]",
                "executive summary",
                [context],
                [step],
                [evidence],
                0.92,
                cited_evidence_ids=["E1"],
                citation_warnings=[],
            ),
            "trace": [{"event": "synthesize", "payload": {"evidence_count": 1}}],
        }


async def test_deepresearch_api_routes_return_run_id_and_list_runs(tmp_path, monkeypatch):
    from api import main

    store = RunTraceStore(base_dir=str(tmp_path))
    monkeypatch.setattr(main, "trace_store", store)
    monkeypatch.setitem(main.workflows, "deepresearch", FakeWorkflow())
    monkeypatch.setitem(main.workflows, "deepsearch", FakeWorkflow())

    request = main.QuestionRequest(question="  What is DeepResearch?  ")
    new_response = await main.deep_research(request)
    legacy_response = await main.deep_search(request)
    rows = await main.list_run_traces(limit=10, workflow="deepresearch")

    assert request.question == "What is DeepResearch?"
    assert new_response.run_id.startswith("deepresearch_")
    assert legacy_response.run_id.startswith("deepresearch_")
    assert new_response.cited_evidence_ids == ["E1"]
    assert len(rows) == 2
    assert all(Path(tmp_path, f"{row.run_id}.json").exists() for row in rows)


async def test_deepresearch_api_persists_failed_run(tmp_path, monkeypatch):
    from api import main

    class BrokenWorkflow:
        async def ainvoke(self, state):
            raise RuntimeError("boom")

    store = RunTraceStore(base_dir=str(tmp_path))
    monkeypatch.setattr(main, "trace_store", store)
    monkeypatch.setitem(main.workflows, "deepresearch", BrokenWorkflow())
    with pytest.raises(HTTPException) as exc_info:
        await main.deep_research(main.QuestionRequest(question="valid question"))
    assert exc_info.value.status_code == 500
    assert store.list_runs()[0]["status"] == "failed"


async def test_upload_sanitizes_filename_and_enforces_limit(tmp_path, monkeypatch):
    from api import main

    monkeypatch.setattr(main.settings, "upload_dir", str(tmp_path))
    monkeypatch.setattr(main.settings, "upload_max_bytes", 4)
    safe_upload = UploadFile(file=io.BytesIO(b"1234"), filename="../safe.txt")
    path, name = await main._save_upload(safe_upload)
    assert name == "safe.txt"
    assert Path(path).parent == tmp_path.resolve()

    large_upload = UploadFile(file=io.BytesIO(b"12345"), filename="safe.txt")
    with pytest.raises(HTTPException) as exc_info:
        await main._save_upload(large_upload)
    assert exc_info.value.status_code == 413
    assert (tmp_path / "safe.txt").read_bytes() == b"1234"
    assert not list(tmp_path.glob("*.upload"))


def test_question_request_rejects_whitespace_only():
    from api.main import QuestionRequest

    with pytest.raises(ValidationError):
        QuestionRequest(question="   ")


def test_real_fastapi_app_starts_without_api_key_or_databases(monkeypatch):
    from api import main

    async def unavailable_init():
        raise ConnectionError("not running")

    monkeypatch.setattr(main.vector_store, "init", unavailable_init)
    monkeypatch.setattr(main.knowledge_graph, "init", unavailable_init)
    with TestClient(main.app) as client:
        response = client.get("/api/health")
        openapi = client.get("/openapi.json")
    assert response.status_code == 200
    assert response.json()["components"]["vector_store"] == "degraded"
    assert openapi.status_code == 200
