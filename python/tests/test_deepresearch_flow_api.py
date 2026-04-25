import asyncio
import sys
import tempfile
import types
import typing
from pathlib import Path


if not hasattr(typing, "Annotated"):
    class _AnnotatedCompat:
        def __getitem__(self, value):
            if isinstance(value, tuple):
                return value[0]
            return value

    typing.Annotated = _AnnotatedCompat()


def _install_dependency_stubs():
    fastapi = types.ModuleType("fastapi")
    fastapi_responses = types.ModuleType("fastapi.responses")
    langchain_core = types.ModuleType("langchain_core")
    langchain_core_messages = types.ModuleType("langchain_core.messages")
    langchain_openai = types.ModuleType("langchain_openai")
    langgraph = types.ModuleType("langgraph")
    langgraph_graph = types.ModuleType("langgraph.graph")
    langgraph_message = types.ModuleType("langgraph.graph.message")
    pydantic = types.ModuleType("pydantic")
    pydantic_settings = types.ModuleType("pydantic_settings")

    class _BaseModel:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)

        def model_dump(self):
            return dict(self.__dict__)

    class _BaseSettings:
        def __init__(self, **kwargs):
            for name, value in self.__class__.__dict__.items():
                if name.startswith("_") or callable(value) or isinstance(value, property):
                    continue
                setattr(self, name, kwargs.get(name, value))

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

    class _HTTPException(Exception):
        def __init__(self, status_code, detail):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    class _FastAPI:
        def __init__(self, *args, **kwargs):
            self.routes = []

        def post(self, *args, **kwargs):
            def _decorator(fn):
                self.routes.append(("POST", args, kwargs, fn))
                return fn

            return _decorator

        def get(self, *args, **kwargs):
            def _decorator(fn):
                self.routes.append(("GET", args, kwargs, fn))
                return fn

            return _decorator

    class _FileResponse:
        def __init__(self, path):
            self.path = path

    class _CompiledGraph:
        def __init__(self, nodes, edges, conditionals, entry):
            self.nodes = nodes
            self.edges = edges
            self.conditionals = conditionals
            self.entry = entry

        async def ainvoke(self, initial_state):
            state = dict(initial_state)
            current = self.entry
            while current != "__end__":
                handler = self.nodes[current]
                update = await handler(state) if asyncio.iscoroutinefunction(handler) else handler(state)
                if update:
                    state.update(update)

                if current in self.conditionals:
                    router, route_map = self.conditionals[current]
                    current = route_map[router(state)]
                    continue

                next_nodes = self.edges.get(current, [])
                current = next_nodes[0] if next_nodes else "__end__"
            return state

    class _StateGraph:
        def __init__(self, *args, **kwargs):
            self.nodes = {}
            self.edges = {}
            self.conditionals = {}
            self.entry = None

        def add_node(self, name, fn):
            self.nodes[name] = fn

        def set_entry_point(self, name):
            self.entry = name

        def add_edge(self, start, end):
            self.edges.setdefault(start, []).append(end)

        def add_conditional_edges(self, start, router, route_map):
            self.conditionals[start] = (router, route_map)

        def compile(self):
            return _CompiledGraph(self.nodes, self.edges, self.conditionals, self.entry)

    fastapi.FastAPI = _FastAPI
    fastapi.File = lambda default=None, **kwargs: default
    fastapi.HTTPException = _HTTPException
    fastapi.Query = lambda default=None, **kwargs: default
    fastapi.UploadFile = object
    fastapi_responses.FileResponse = _FileResponse
    langchain_core_messages.HumanMessage = _Message
    langchain_core_messages.SystemMessage = _Message
    langchain_openai.ChatOpenAI = _ChatOpenAI
    langchain_openai.OpenAIEmbeddings = _OpenAIEmbeddings
    langgraph_graph.END = "__end__"
    langgraph_graph.StateGraph = _StateGraph
    langgraph_message.add_messages = lambda left, right: (left or []) + (right or [])
    pydantic.BaseModel = _BaseModel
    pydantic_settings.BaseSettings = _BaseSettings

    sys.modules.setdefault("fastapi", fastapi)
    sys.modules.setdefault("fastapi.responses", fastapi_responses)
    sys.modules.setdefault("langchain_core", langchain_core)
    sys.modules.setdefault("langchain_core.messages", langchain_core_messages)
    sys.modules.setdefault("langchain_openai", langchain_openai)
    sys.modules.setdefault("langgraph", langgraph)
    sys.modules.setdefault("langgraph.graph", langgraph_graph)
    sys.modules.setdefault("langgraph.graph.message", langgraph_message)
    sys.modules.setdefault("pydantic", pydantic)
    sys.modules.setdefault("pydantic_settings", pydantic_settings)


_install_dependency_stubs()

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

    async def plan_search(self, question):
        return {"goal": question, "initial_queries": ["initial query"], "expected_evidence": ["web pages"]}

    def initial_frontier(self, plan, question):
        return list(plan["initial_queries"])

    async def retrieve_round(self, queries):
        self.retrieve_calls.append(list(queries))
        return [
            RetrievedContext(
                content=f"Evidence collected for {queries[0]}",
                source=f"https://example.com/{len(self.retrieve_calls)}",
                score=0.8 + len(self.retrieve_calls) / 100,
                retrieval_type="web_page",
                metadata={"title": f"Page {len(self.retrieve_calls)}"},
            )
        ]

    def merge_contexts(self, contexts):
        return contexts

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
        return DeepResearchStep(
            iteration=iteration,
            focus=focus,
            queries=queries,
            summary=summary,
            gaps=gaps,
            contexts=contexts,
        )

    @staticmethod
    def build_evidence(contexts):
        return [
            DeepResearchEvidence(
                evidence_id=f"E{index}",
                source=context.source,
                title=context.metadata.get("title", context.source),
                quote=context.content,
                confidence=context.score,
                retrieved_at="2026-04-24T00:00:00+00:00",
            )
            for index, context in enumerate(contexts, start=1)
        ]

    async def compose_answer(self, question, plan, steps, contexts, evidence, fallback_confidence):
        assert evidence and evidence[0].evidence_id == "E1"
        return "summary", "answer with citation [E1]", 0.91

    @staticmethod
    def build_result(question, answer, executive_summary, contexts, steps, evidence, confidence):
        return DeepResearchResult(
            question=question,
            answer=answer,
            executive_summary=executive_summary,
            contexts=contexts,
            steps=steps,
            evidence=evidence,
            confidence=confidence,
        )


def test_deepresearch_graph_runs_multiple_rounds_and_synthesizes_evidence():
    fake_agent = FakeDeepResearchAgent()
    graph = _build_deepresearch_graph(fake_agent)

    result_state = asyncio.run(graph.ainvoke({"question": "How should we build DeepResearch?"}))

    assert fake_agent.retrieve_calls == [["initial query"], ["follow up query"]]
    assert result_state["result"].iterations == 2
    assert result_state["result"].answer == "answer with citation [E1]"
    assert result_state["result"].evidence[0].evidence_id == "E1"
    assert [item["event"] for item in result_state["trace"]] == [
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
            content="Fetched article body",
            source="https://example.com/research",
            score=0.92,
            retrieval_type="web_page",
            metadata={"title": "Research Page"},
        )
        evidence = DeepResearchEvidence(
            evidence_id="E1",
            source=context.source,
            title="Research Page",
            quote="Fetched article body",
            confidence=0.92,
            retrieved_at="2026-04-24T00:00:00+00:00",
        )
        step = DeepResearchStep(
            iteration=1,
            focus="research focus",
            queries=["deepresearch query"],
            summary="round summary",
            gaps=[],
            contexts=[context],
        )
        return {
            "result": DeepResearchResult(
                question=state["question"],
                answer="final answer [E1]",
                executive_summary="executive summary",
                contexts=[context],
                steps=[step],
                evidence=[evidence],
                confidence=0.92,
            ),
            "trace": [{"event": "synthesize", "payload": {"evidence_count": 1}}],
        }


def test_deepresearch_api_new_and_legacy_routes_return_run_id_and_list_runs():
    from api import main

    with tempfile.TemporaryDirectory() as tmpdir:
        main.trace_store = RunTraceStore(base_dir=tmpdir)
        main.workflows.clear()
        main.workflows.update(
            {
                "deepresearch": FakeWorkflow(),
                "deepsearch": FakeWorkflow(),
            }
        )

        req = main.QuestionRequest(question="What is DeepResearch?")
        new_response = asyncio.run(main.deep_research(req))
        legacy_response = asyncio.run(main.deep_search(req))
        rows = asyncio.run(main.list_run_traces(limit=10, workflow="deepresearch"))

        assert new_response.run_id.startswith("deepresearch_")
        assert legacy_response.run_id.startswith("deepresearch_")
        assert new_response.evidence[0].evidence_id == "E1"
        assert new_response.sources[0]["type"] == "web_page"
        assert len(rows) == 2
        assert {row.run_id for row in rows} == {new_response.run_id, legacy_response.run_id}
        assert all(Path(tmpdir, f"{row.run_id}.json").exists() for row in rows)
