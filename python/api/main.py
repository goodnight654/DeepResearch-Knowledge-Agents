"""
FastAPI 入口 — DeepResearch Knowledge Hub REST API

提供三组接口:
  1. /api/ingest   — 文档上传 & 入库
  2. /api/qa       — 智能问答 / DeepResearch
  3. /api/admin    — 管理（统计、更新触发）
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from agents.doc_parser_agent import DocParserAgent
from agents.knowledge_update_agent import ChangeType, DocumentChange
from config import settings
from orchestrator.graph import build_knowledge_graph_workflow
from services.knowledge_graph import KnowledgeGraphService
from services.run_trace_store import RunTraceStore
from services.vector_store import VectorStoreService
from services.web_page_reader import WebPageReader
from services.web_search import WebSearchService

logger = logging.getLogger(__name__)

vector_store = VectorStoreService()
knowledge_graph = KnowledgeGraphService()
web_search = WebSearchService()
web_page_reader = WebPageReader()
trace_store = RunTraceStore()
workflows: dict[str, Any] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(settings.upload_dir, exist_ok=True)

    async def initialize(name: str, initializer: Any) -> None:
        try:
            await asyncio.wait_for(initializer(), timeout=settings.dependency_init_timeout_seconds)
        except Exception as exc:
            logger.warning("%s unavailable; retrieval will degrade gracefully: %s", name, exc)

    await asyncio.gather(
        initialize("vector store", vector_store.init),
        initialize("knowledge graph", knowledge_graph.init),
    )
    workflows.update(
        build_knowledge_graph_workflow(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            web_search=web_search,
            web_page_reader=web_page_reader,
        )
    )
    yield
    await knowledge_graph.close()


app = FastAPI(
    title="DeepResearch Knowledge Hub",
    description="本地知识库 + Web Search + evidence-level citations 的深度研究 Agent API",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Request / Response Models ────────────────────────────────


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8000)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        normalized = " ".join(value.split()).strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized


class QuestionResponse(BaseModel):
    run_id: str
    question: str
    answer: str
    confidence: float
    intent: str
    sources: list[dict[str, Any]]
    reasoning_steps: list[str]


class DeepResearchStepResponse(BaseModel):
    iteration: int
    focus: str
    queries: list[str]
    summary: str
    gaps: list[str]
    contexts_count: int


class EvidenceResponse(BaseModel):
    evidence_id: str
    source: str
    title: str
    quote: str
    confidence: float
    retrieved_at: str


class DeepResearchResponse(BaseModel):
    run_id: str
    question: str
    executive_summary: str
    answer: str
    confidence: float
    iterations: int
    steps: list[DeepResearchStepResponse]
    evidence: list[EvidenceResponse]
    cited_evidence_ids: list[str]
    citation_warnings: list[str]
    sources: list[dict[str, Any]]


DeepSearchStepResponse = DeepResearchStepResponse
DeepSearchResponse = DeepResearchResponse


class IngestResponse(BaseModel):
    file_name: str
    chunks_count: int
    entities_count: int
    relations_count: int
    status: str


class StatsResponse(BaseModel):
    vector_store: dict[str, Any]
    knowledge_graph: dict[str, Any]


class UpdateRequest(BaseModel):
    file_path: str = Field(min_length=1)
    change_type: ChangeType = ChangeType.MODIFIED


class UpdateResponse(BaseModel):
    file_path: str
    vectors_added: int
    vectors_deleted: int
    entities_added: int
    relations_added: int
    success: bool
    processing_time_ms: float


class RunTraceResponse(BaseModel):
    run_id: str
    workflow: str
    status: str
    created_at: str
    input: dict[str, Any]
    trace: list[dict[str, Any]]
    output: dict[str, Any]
    metadata: dict[str, Any]
    error: str | None = None


class RunTraceSummaryResponse(BaseModel):
    run_id: str
    workflow: str
    status: str
    created_at: str
    question: str
    confidence: float | None = None
    iterations: int | None = None


# ── Ingest Endpoints ─────────────────────────────────────────


@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["文档入库"])
async def upload_document(file: Annotated[UploadFile, File()]):
    """上传并解析文档，自动入库到向量库和知识图谱"""
    ingest_wf = workflows.get("ingest")
    if not ingest_wf:
        raise HTTPException(status_code=503, detail="Ingest workflow not initialized")
    save_path, safe_name = await _save_upload(file)

    result = await ingest_wf.ainvoke({"file_paths": [save_path]})
    chunks = result.get("chunks", [])
    extractions = result.get("extractions", [])
    total_entities = sum(len(e.entities) for e in extractions)
    total_relations = sum(len(e.relations) for e in extractions)

    return IngestResponse(
        file_name=safe_name,
        chunks_count=len(chunks),
        entities_count=total_entities,
        relations_count=total_relations,
        status="success",
    )


@app.post("/api/ingest/batch", response_model=list[IngestResponse], tags=["文档入库"])
async def upload_batch(files: Annotated[list[UploadFile], File()]):
    """批量上传文档"""
    return await asyncio.gather(*(upload_document(file) for file in files))


async def _save_upload(file: UploadFile) -> tuple[str, str]:
    """Persist an upload within the configured directory and enforce a size limit."""
    safe_name = Path(file.filename or "upload.bin").name
    if safe_name in {"", ".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid file name")
    if Path(safe_name).suffix.lower() not in DocParserAgent.SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="Unsupported document type")
    upload_root = Path(settings.upload_dir).expanduser().resolve()
    upload_root.mkdir(parents=True, exist_ok=True)
    destination = (upload_root / safe_name).resolve()
    if destination.parent != upload_root:
        raise HTTPException(status_code=400, detail="Invalid file name")
    temporary = upload_root / f".{safe_name}.{uuid4().hex}.upload"

    written = 0
    try:
        with temporary.open("wb") as output:
            while chunk := await file.read(1024 * 1024):
                written += len(chunk)
                if written > settings.upload_max_bytes:
                    raise HTTPException(status_code=413, detail="Uploaded file is too large")
                output.write(chunk)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return str(destination), safe_name


# ── QA Endpoints ─────────────────────────────────────────────


@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["智能问答"])
async def ask_question(req: QuestionRequest):
    """智能问答 — 混合检索 + 知识图谱推理"""
    qa_wf = workflows.get("qa")
    if not qa_wf:
        raise HTTPException(status_code=503, detail="QA workflow not initialized")

    result = await qa_wf.ainvoke({"question": req.question})
    qa_result = result.get("result")
    if not qa_result:
        raise HTTPException(status_code=500, detail="QA failed")

    trace = [
        {
            "ts": datetime.now(timezone.utc).isoformat(),
            "event": "qa_answer",
            "payload": {
                "intent": qa_result.intent.value,
                "confidence": qa_result.confidence,
                "sources_count": len(qa_result.contexts),
                "reasoning_steps": qa_result.reasoning_steps,
            },
        }
    ]
    run_id = trace_store.save_run(
        workflow="qa",
        input_payload={"question": req.question},
        trace=trace,
        output={
            "question": qa_result.question,
            "answer": qa_result.answer,
            "confidence": qa_result.confidence,
            "intent": qa_result.intent.value,
        },
    )

    return QuestionResponse(
        run_id=run_id,
        question=qa_result.question,
        answer=qa_result.answer,
        confidence=qa_result.confidence,
        intent=qa_result.intent.value,
        sources=[
            {"content": c.content[:200], "source": c.source, "score": c.score, "type": c.retrieval_type}
            for c in qa_result.contexts
        ],
        reasoning_steps=qa_result.reasoning_steps,
    )


@app.post("/api/qa/deep-research", response_model=DeepResearchResponse, tags=["智能问答"])
async def deep_research(req: QuestionRequest):
    """复杂问题 DeepResearch — 计划、网页阅读、证据归因、最终综合回答"""
    return await _run_deep_research(req, workflow_key="deepresearch", workflow_name="deepresearch")


@app.post("/api/qa/deep-search", response_model=DeepResearchResponse, tags=["智能问答"])
async def deep_search(req: QuestionRequest):
    """兼容旧入口：复杂问题 DeepResearch"""
    return await _run_deep_research(req, workflow_key="deepsearch", workflow_name="deepresearch")


async def _run_deep_research(
    req: QuestionRequest,
    workflow_key: str,
    workflow_name: str,
) -> DeepResearchResponse:
    deepresearch_wf = workflows.get(workflow_key)
    if not deepresearch_wf:
        raise HTTPException(status_code=503, detail="DeepResearch workflow not initialized")

    try:
        result = await deepresearch_wf.ainvoke({"question": req.question})
    except Exception as exc:
        run_id = trace_store.save_run(
            workflow=workflow_name,
            input_payload={"question": req.question},
            trace=[],
            output={"question": req.question, "confidence": 0.0, "iterations": 0},
            status="failed",
            error=str(exc),
        )
        logger.exception("DeepResearch run %s failed", run_id)
        raise HTTPException(status_code=500, detail=f"DeepResearch failed; run_id={run_id}") from exc
    search_result = result.get("result")
    if not search_result:
        raise HTTPException(status_code=500, detail="DeepResearch failed")
    trace = result.get("trace", [])
    run_id = trace_store.save_run(
        workflow=workflow_name,
        input_payload={"question": req.question},
        trace=trace if isinstance(trace, list) else [],
        output={
            "question": search_result.question,
            "executive_summary": search_result.executive_summary,
            "answer": search_result.answer,
            "confidence": search_result.confidence,
            "iterations": search_result.iterations,
            "evidence_count": len(search_result.evidence),
        },
    )

    return DeepResearchResponse(
        run_id=run_id,
        question=search_result.question,
        executive_summary=search_result.executive_summary,
        answer=search_result.answer,
        confidence=search_result.confidence,
        iterations=search_result.iterations,
        steps=[
            DeepResearchStepResponse(
                iteration=step.iteration,
                focus=step.focus,
                queries=step.queries,
                summary=step.summary,
                gaps=step.gaps,
                contexts_count=len(step.contexts),
            )
            for step in search_result.steps
        ],
        evidence=[
            EvidenceResponse(
                evidence_id=item.evidence_id,
                source=item.source,
                title=item.title,
                quote=item.quote,
                confidence=item.confidence,
                retrieved_at=item.retrieved_at,
            )
            for item in search_result.evidence
        ],
        cited_evidence_ids=search_result.cited_evidence_ids,
        citation_warnings=search_result.citation_warnings,
        sources=[
            {
                "content": ctx.content[:200],
                "source": ctx.source,
                "score": ctx.score,
                "type": ctx.retrieval_type,
            }
            for ctx in search_result.contexts
        ],
    )


@app.get("/api/qa/runs/{run_id}", response_model=RunTraceResponse, tags=["智能问答"])
async def get_run_trace(run_id: str):
    """查询某次 QA / DeepResearch 运行轨迹"""
    record = trace_store.load_run(run_id)
    if not record:
        raise HTTPException(status_code=404, detail="Run trace not found")
    return RunTraceResponse(
        run_id=record.get("run_id", run_id),
        workflow=record.get("workflow", "unknown"),
        status=record.get("status", "unknown"),
        created_at=record.get("created_at", ""),
        input=record.get("input", {}),
        trace=record.get("trace", []),
        output=record.get("output", {}),
        metadata=record.get("metadata", {}),
        error=record.get("error"),
    )


@app.get("/api/qa/runs", response_model=list[RunTraceSummaryResponse], tags=["智能问答"])
async def list_run_traces(
    limit: int = Query(default=20, ge=1, le=200),
    workflow: str | None = Query(default=None),
):
    """列出最近运行记录，供前端 trace viewer 展示"""
    rows = trace_store.list_runs(limit=limit, workflow=workflow)
    return [
        RunTraceSummaryResponse(
            run_id=str(row.get("run_id", "")),
            workflow=str(row.get("workflow", "unknown")),
            status=str(row.get("status", "unknown")),
            created_at=str(row.get("created_at", "")),
            question=str(row.get("question", "")),
            confidence=(float(row.get("confidence")) if row.get("confidence") is not None else None),
            iterations=(int(row.get("iterations")) if row.get("iterations") is not None else None),
        )
        for row in rows
    ]


@app.get("/ui/trace", include_in_schema=False)
async def trace_viewer_page():
    """Trace 可视化页面"""
    page_path = os.path.join(os.path.dirname(__file__), "static", "trace-viewer.html")
    if not os.path.exists(page_path):
        raise HTTPException(status_code=404, detail="Trace viewer page not found")
    return FileResponse(page_path)


# ── Admin Endpoints ──────────────────────────────────────────


@app.get("/api/admin/stats", response_model=StatsResponse, tags=["系统管理"])
async def get_stats():
    """获取系统统计信息"""
    vs_stats = await vector_store.get_stats()
    kg_stats = await knowledge_graph.get_stats()
    return StatsResponse(vector_store=vs_stats, knowledge_graph=kg_stats)


@app.post("/api/admin/update", response_model=UpdateResponse, tags=["系统管理"])
async def trigger_update(req: UpdateRequest):
    """手动触发知识更新"""
    update_wf = workflows.get("update")
    if not update_wf:
        raise HTTPException(status_code=503, detail="Update workflow not initialized")

    change = DocumentChange(
        file_path=req.file_path,
        change_type=req.change_type,
    )
    result = await update_wf.ainvoke({"changes": [change]})
    results = result.get("results", [])
    if not results:
        raise HTTPException(status_code=500, detail="Update failed")

    r = results[0]
    return UpdateResponse(
        file_path=r.change.file_path,
        vectors_added=r.vectors_added,
        vectors_deleted=r.vectors_deleted,
        entities_added=r.entities_added,
        relations_added=r.relations_added,
        success=r.success,
        processing_time_ms=r.processing_time_ms,
    )


@app.get("/api/health", tags=["系统管理"])
async def health():
    return {
        "status": "ok",
        "service": "DeepResearch Knowledge Hub",
        "components": {
            "vector_store": "ready" if vector_store.available else "degraded",
            "knowledge_graph": "ready" if knowledge_graph.available else "degraded",
            "web_search": "ready" if web_search.enabled else "disabled",
            "llm_configured": bool(settings.openai_api_key.strip()),
        },
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
