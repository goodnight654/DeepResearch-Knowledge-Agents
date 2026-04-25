"""
LangGraph 编排引擎 — 5 Agent 混合编排

编排模式:
  1. 文档入库流程: DocParser → KnowledgeExtract → (VectorStore + KnowledgeGraph)
  2. 问答流程: Query → QA Agent → (VectorRetrieval ∥ GraphRetrieval) → Answer
  3. DeepResearch 流程: Query → Plan → Multi-Round Retrieval → Gap Analysis → Synthesis
  4. 增量更新流程: CDC Event → UpdateAgent → (Diff → Parse → Store)

使用 LangGraph StateGraph 实现有向图编排，支持条件路由和并行分支
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Any

from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from agents.deepresearch_agent import DeepResearchAgent, DeepResearchResult
from agents.doc_parser_agent import DocParserAgent, DocumentChunk
from agents.knowledge_extract_agent import ExtractionResult, KnowledgeExtractAgent
from agents.knowledge_update_agent import (
    DocumentChange,
    KnowledgeUpdateAgent,
    UpdateResult,
)
from agents.qa_agent import QAAgent, QAResult
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService


class WorkflowType(str, Enum):
    INGEST = "ingest"
    QA = "qa"
    DEEPRESEARCH = "deepresearch"
    DEEPSEARCH = "deepsearch"
    UPDATE = "update"


# ── State Schemas ────────────────────────────────────────────

class IngestState(dict):
    """文档入库流程状态"""
    file_paths: list[str]
    chunks: list[DocumentChunk]
    extractions: list[ExtractionResult]
    vectors_stored: int
    entities_stored: int
    messages: Annotated[list, add_messages]


class QAState(dict):
    """问答流程状态"""
    question: str
    result: QAResult | None
    messages: Annotated[list, add_messages]


class DeepResearchState(dict):
    """DeepResearch 流程状态"""
    question: str
    result: DeepResearchResult | None
    messages: Annotated[list, add_messages]


class DeepSearchState(DeepResearchState):
    """Backward-compatible DeepSearch state alias."""


class UpdateState(dict):
    """增量更新流程状态"""
    changes: list[DocumentChange]
    results: list[UpdateResult]
    messages: Annotated[list, add_messages]


# ── Workflow Builder ─────────────────────────────────────────

def build_knowledge_graph_workflow(
    vector_store: VectorStoreService | None = None,
    knowledge_graph: KnowledgeGraphService | None = None,
    web_search: Any = None,
    web_page_reader: Any = None,
) -> dict[str, Any]:
    """
    构建四条编排流水线，返回 {"ingest": graph, "qa": graph, "deepresearch": graph, "deepsearch": graph, "update": graph}
    """
    doc_parser = DocParserAgent()
    extractor = KnowledgeExtractAgent()
    qa_agent = QAAgent(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        web_search=web_search,
        web_page_reader=web_page_reader,
    )
    deepresearch_agent = DeepResearchAgent(
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
        web_search=web_search,
        web_page_reader=web_page_reader,
    )
    update_agent = KnowledgeUpdateAgent(
        doc_parser=doc_parser,
        knowledge_extractor=extractor,
        vector_store=vector_store,
        knowledge_graph=knowledge_graph,
    )

    return {
        "ingest": _build_ingest_graph(doc_parser, extractor, vector_store, knowledge_graph),
        "qa": _build_qa_graph(qa_agent),
        "deepresearch": _build_deepresearch_graph(deepresearch_agent),
        "deepsearch": _build_deepresearch_graph(deepresearch_agent),
        "update": _build_update_graph(update_agent),
    }


# ── Ingest Pipeline ─────────────────────────────────────────

def _build_ingest_graph(
    doc_parser: DocParserAgent,
    extractor: KnowledgeExtractAgent,
    vector_store: VectorStoreService | None,
    knowledge_graph: KnowledgeGraphService | None,
) -> StateGraph:

    async def parse_documents(state: dict) -> dict:
        file_paths = state.get("file_paths", [])
        chunks = await doc_parser.parse_batch(file_paths)
        return {"chunks": chunks}

    async def extract_knowledge(state: dict) -> dict:
        chunks = state.get("chunks", [])
        extractions = await extractor.extract(chunks)
        return {"extractions": extractions}

    async def store_vectors(state: dict) -> dict:
        chunks = state.get("chunks", [])
        count = 0
        if vector_store and chunks:
            count = await vector_store.add_chunks(chunks)
        return {"vectors_stored": count}

    async def store_graph(state: dict) -> dict:
        extractions = state.get("extractions", [])
        entity_count = 0
        if knowledge_graph:
            for ext in extractions:
                for ent in ext.entities:
                    await knowledge_graph.upsert_entity(ent)
                    entity_count += 1
                for rel in ext.relations:
                    await knowledge_graph.add_relation(rel)
        return {"entities_stored": entity_count}

    graph = StateGraph(dict)
    graph.add_node("parse", parse_documents)
    graph.add_node("extract", extract_knowledge)
    graph.add_node("store_vectors", store_vectors)
    graph.add_node("store_graph", store_graph)

    graph.set_entry_point("parse")
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", "store_vectors")
    graph.add_edge("extract", "store_graph")
    graph.add_edge("store_vectors", END)
    graph.add_edge("store_graph", END)

    return graph.compile()


# ── QA Pipeline ──────────────────────────────────────────────

def _build_qa_graph(qa_agent: QAAgent) -> StateGraph:

    async def process_question(state: dict) -> dict:
        question = state.get("question", "")
        result = await qa_agent.answer(question)
        return {"result": result}

    graph = StateGraph(dict)
    graph.add_node("answer", process_question)
    graph.set_entry_point("answer")
    graph.add_edge("answer", END)

    return graph.compile()


# ── DeepResearch Pipeline ───────────────────────────────────

def _build_deepresearch_graph(deepresearch_agent: DeepResearchAgent) -> StateGraph:
    def _trace_event(event: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }

    async def plan_search(state: dict) -> dict:
        question = state.get("question", "")
        plan = await deepresearch_agent.plan_search(question)
        focus = str(plan.get("goal") or question)
        frontier = deepresearch_agent.initial_frontier(plan, question)
        trace = state.get("trace", []) + [
            _trace_event(
                "plan",
                {
                    "goal": plan.get("goal", ""),
                    "sub_questions": plan.get("sub_questions", []),
                    "initial_queries": frontier,
                },
            )
        ]
        return {
            "plan": plan,
            "focus": focus,
            "frontier": frontier,
            "iteration": 0,
            "all_contexts": [],
            "steps": [],
            "confidence": 0.0,
            "trace": trace,
        }

    async def retrieve_round(state: dict) -> dict:
        question = state.get("question", "")
        iteration = int(state.get("iteration", 0)) + 1
        frontier = state.get("frontier", []) or [question]
        queries = frontier[: deepresearch_agent.queries_per_iteration] or [question]
        round_contexts = await deepresearch_agent.retrieve_round(queries)
        all_contexts = deepresearch_agent.merge_contexts(state.get("all_contexts", []) + round_contexts)
        trace = state.get("trace", []) + [
            _trace_event(
                "retrieve_round",
                {
                    "iteration": iteration,
                    "queries": queries,
                    "round_contexts_count": len(round_contexts),
                    "total_contexts_count": len(all_contexts),
                },
            )
        ]
        return {
            "iteration": iteration,
            "current_queries": queries,
            "current_round_contexts": round_contexts,
            "all_contexts": all_contexts,
            "trace": trace,
        }

    async def summarize_round(state: dict) -> dict:
        question = state.get("question", "")
        focus = str(state.get("focus", question))
        current_queries = state.get("current_queries", [])
        all_contexts = state.get("all_contexts", [])
        summary = await deepresearch_agent.summarize_round(question, focus, current_queries, all_contexts)
        trace = state.get("trace", []) + [
            _trace_event(
                "summarize_round",
                {
                    "iteration": int(state.get("iteration", 0)),
                    "focus": focus,
                    "summary_preview": summary[:300],
                },
            )
        ]
        return {"current_summary": summary, "trace": trace}

    async def assess_gap(state: dict) -> dict:
        question = state.get("question", "")
        focus = str(state.get("focus", question))
        steps = state.get("steps", [])
        summary = str(state.get("current_summary", ""))
        all_contexts = state.get("all_contexts", [])

        gap_state = await deepresearch_agent.analyze_gaps(
            question=question,
            focus=focus,
            steps=steps,
            summary=summary,
            contexts=all_contexts,
        )
        follow_up = deepresearch_agent.clean_list(gap_state.get("follow_up_queries"))
        should_finalize = deepresearch_agent.should_finalize(
            iteration=int(state.get("iteration", 0)),
            gap_state=gap_state,
            follow_up_queries=follow_up,
        )

        step = deepresearch_agent.make_step(
            iteration=int(state.get("iteration", 0)),
            focus=focus,
            queries=state.get("current_queries", []),
            summary=summary,
            gaps=deepresearch_agent.clean_list(gap_state.get("gaps")),
            contexts=state.get("current_round_contexts", []),
        )
        updated_steps = steps + [step]
        next_focus = str(gap_state.get("next_focus") or focus)
        confidence = float(gap_state.get("confidence", state.get("confidence", 0.0)))
        trace = state.get("trace", []) + [
            _trace_event(
                "assess_gap",
                {
                    "iteration": int(state.get("iteration", 0)),
                    "answered": bool(gap_state.get("answered")),
                    "should_finalize": should_finalize,
                    "next_focus": next_focus,
                    "follow_up_queries": follow_up,
                    "gaps": deepresearch_agent.clean_list(gap_state.get("gaps")),
                    "confidence": confidence,
                },
            )
        ]

        return {
            "steps": updated_steps,
            "frontier": follow_up,
            "focus": next_focus,
            "confidence": confidence,
            "should_finalize": should_finalize,
            "trace": trace,
        }

    def route_after_gap(state: dict) -> str:
        if state.get("should_finalize"):
            return "synthesize"
        return "retrieve"

    async def synthesize(state: dict) -> dict:
        question = state.get("question", "")
        plan = state.get("plan", {})
        steps = state.get("steps", [])
        contexts = deepresearch_agent.merge_contexts(state.get("all_contexts", []))[: deepresearch_agent.final_context_limit]
        evidence = deepresearch_agent.build_evidence(contexts)

        executive_summary, answer, confidence = await deepresearch_agent.compose_answer(
            question=question,
            plan=plan,
            steps=steps,
            contexts=contexts,
            evidence=evidence,
            fallback_confidence=float(state.get("confidence", 0.0)),
        )
        result = deepresearch_agent.build_result(
            question=question,
            answer=answer,
            executive_summary=executive_summary,
            contexts=contexts,
            steps=steps,
            evidence=evidence,
            confidence=confidence,
        )
        trace = state.get("trace", []) + [
            _trace_event(
                "synthesize",
                {
                    "iterations": len(steps),
                    "contexts_count": len(contexts),
                    "confidence": confidence,
                },
            )
        ]
        return {"result": result, "trace": trace}

    graph = StateGraph(dict)
    graph.add_node("plan_search", plan_search)
    graph.add_node("retrieve_round", retrieve_round)
    graph.add_node("summarize_round", summarize_round)
    graph.add_node("assess_gap", assess_gap)
    graph.add_node("synthesize", synthesize)

    graph.set_entry_point("plan_search")
    graph.add_edge("plan_search", "retrieve_round")
    graph.add_edge("retrieve_round", "summarize_round")
    graph.add_edge("summarize_round", "assess_gap")
    graph.add_conditional_edges(
        "assess_gap",
        route_after_gap,
        {"retrieve": "retrieve_round", "synthesize": "synthesize"},
    )
    graph.add_edge("synthesize", END)

    return graph.compile()


# ── Update Pipeline ──────────────────────────────────────────

def _build_update_graph(update_agent: KnowledgeUpdateAgent) -> StateGraph:

    async def process_updates(state: dict) -> dict:
        changes = state.get("changes", [])
        results = await update_agent.process_batch(changes)
        return {"results": results}

    def should_continue(state: dict) -> str:
        results = state.get("results", [])
        failed = [r for r in results if not r.success]
        if failed:
            return "retry"
        return "done"

    async def retry_failed(state: dict) -> dict:
        results = state.get("results", [])
        failed_changes = [r.change for r in results if not r.success]
        retried = await update_agent.process_batch(failed_changes)
        all_results = [r for r in results if r.success] + retried
        return {"results": all_results}

    graph = StateGraph(dict)
    graph.add_node("process", process_updates)
    graph.add_node("retry", retry_failed)

    graph.set_entry_point("process")
    graph.add_conditional_edges("process", should_continue, {"retry": "retry", "done": END})
    graph.add_edge("retry", END)

    return graph.compile()
