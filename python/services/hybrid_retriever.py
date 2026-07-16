"""
统一混合检索服务 — 查询改写 + 向量检索 + 图谱检索 + 重排序

职责:
  1. 查询分析（改写 / 实体抽取 / 关键词提取）
  2. 向量语义检索
  3. 图谱实体 / 子图 / Cypher 检索
  4. 网页正文读取与证据级上下文构造
  5. 跨来源结果去重与重排序
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from utils.json_utils import coerce_float, parse_json_object
from utils.openai_clients import create_chat_model

logger = logging.getLogger(__name__)


@dataclass
class QueryAnalysis:
    question: str
    queries: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)


@dataclass
class RetrievedContext:
    content: str
    source: str
    score: float
    retrieval_type: str  # "vector" | "graph" | "web" | "web_page" | "hybrid"
    metadata: dict[str, Any] = field(default_factory=dict)


QUERY_REWRITE_PROMPT = """\
你是一个查询改写专家。将用户问题改写为更适合检索的形式。
要求：
1. 提取核心实体和关键词
2. 生成 1-3 个检索查询
3. 返回 JSON: {"queries": ["查询1", "查询2"], "entities": ["实体1"], "keywords": ["关键词1"]}
只返回 JSON。
"""

CYPHER_GENERATION_PROMPT = """\
你是一个 Neo4j Cypher 查询生成专家。根据用户问题和提取的实体，生成 Cypher 查询。

知识图谱 Schema:
- 节点标签: Entity
- 常见属性: name, type, description, source, version
- 关系通过知识抽取动态生成

返回 JSON: {"queries": ["MATCH ... RETURN ...", "MATCH ... RETURN ..."]}
只返回 JSON，不要其他文字。
"""


class HybridRetriever:
    """问答与 DeepResearch 共用的统一检索层。"""

    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
        web_search: Any = None,
        web_page_reader: Any = None,
        llm: Any = None,
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.web_search = web_search
        self.web_page_reader = web_page_reader
        self.llm = llm or create_chat_model()

    async def analyze(self, question: str) -> QueryAnalysis:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")
        data = await self._load_json_response(
            QUERY_REWRITE_PROMPT,
            question,
            fallback={"queries": [question], "entities": [], "keywords": []},
        )
        raw_queries = data.get("queries", [])
        raw_entities = data.get("entities", [])
        raw_keywords = data.get("keywords", [])
        queries = (
            [q.strip() for q in raw_queries if isinstance(q, str) and q.strip()]
            if isinstance(raw_queries, list)
            else []
        )
        if question not in queries:
            queries.insert(0, question)
        return QueryAnalysis(
            question=question,
            queries=list(dict.fromkeys(queries))[:3],
            entities=(
                list(dict.fromkeys(e.strip() for e in raw_entities if isinstance(e, str) and e.strip()))[:6]
                if isinstance(raw_entities, list)
                else []
            ),
            keywords=(
                list(dict.fromkeys(k.strip() for k in raw_keywords if isinstance(k, str) and k.strip()))[:10]
                if isinstance(raw_keywords, list)
                else []
            ),
        )

    async def retrieve(
        self,
        question: str,
        top_k: int = 8,
        per_query_limit: int = 5,
        web_limit: int | None = None,
    ) -> tuple[QueryAnalysis, list[RetrievedContext]]:
        top_k = max(1, min(int(top_k), 100))
        per_query_limit = max(1, min(int(per_query_limit), 50))
        analysis = await self.analyze(question)
        results = await asyncio.gather(
            self._vector_retrieve(analysis, per_query_limit),
            self._graph_retrieve(question, analysis, per_query_limit),
            self._web_retrieve(analysis, web_limit or min(top_k, settings.web_search_top_k)),
            return_exceptions=True,
        )
        labels = ("vector", "graph", "web")
        collected: list[list[RetrievedContext]] = []
        for label, result in zip(labels, results, strict=False):
            if isinstance(result, Exception):
                logger.warning("%s retrieval failed: %s", label, result)
                collected.append([])
            else:
                collected.append(result)
        vector_contexts, graph_contexts, web_contexts = collected
        contexts = self._hybrid_rerank(vector_contexts + graph_contexts + web_contexts)
        return analysis, contexts[:top_k]

    async def _vector_retrieve(
        self,
        analysis: QueryAnalysis,
        per_query_limit: int,
    ) -> list[RetrievedContext]:
        if not self.vector_store:
            return []

        contexts: list[RetrievedContext] = []
        for query in analysis.queries:
            try:
                results = await self.vector_store.search(query, top_k=per_query_limit)
            except Exception as exc:
                logger.warning("vector query failed for %r: %s", query, exc)
                continue
            for doc, score in results:
                if not isinstance(doc, dict):
                    continue
                content = str(doc.get("content", "") or "").strip()
                if not content:
                    continue
                metadata = doc.get("metadata", {})
                if not isinstance(metadata, dict):
                    metadata = {}
                contexts.append(
                    RetrievedContext(
                        content=content,
                        source=str(doc.get("source", "vector_store") or "vector_store"),
                        score=coerce_float(score),
                        retrieval_type="vector",
                        metadata={
                            **metadata,
                            "query": query,
                        },
                    )
                )
        return contexts

    async def _graph_retrieve(
        self,
        question: str,
        analysis: QueryAnalysis,
        per_query_limit: int,
    ) -> list[RetrievedContext]:
        if not self.knowledge_graph:
            return []

        contexts: list[RetrievedContext] = []

        for entity in analysis.entities[:3]:
            try:
                entity_hits = await self.knowledge_graph.search_entities(entity, limit=3)
                neighbors = await self.knowledge_graph.get_neighbors(entity, hops=2)
            except Exception as exc:
                logger.warning("graph lookup failed for %r: %s", entity, exc)
                continue
            for hit in entity_hits:
                if not isinstance(hit, dict):
                    continue
                contexts.append(
                    RetrievedContext(
                        content=(
                            f"实体: {hit.get('name', '')} | 类型: {hit.get('type', '')} | "
                            f"描述: {hit.get('description', '')}"
                        ),
                        source="knowledge_graph",
                        score=0.72,
                        retrieval_type="graph",
                        metadata={"query": entity, "kind": "entity_search"},
                    )
                )

            for record in neighbors[:per_query_limit]:
                if not isinstance(record, dict):
                    continue
                contexts.append(
                    RetrievedContext(
                        content=(
                            f"{record.get('source', '')} --[{', '.join(record.get('relations', []))}]--> "
                            f"{record.get('target', '')} ({record.get('target_type', '')}): "
                            f"{record.get('target_desc', '')}"
                        ),
                        source="knowledge_graph",
                        score=0.78,
                        retrieval_type="graph",
                        metadata={"query": entity, "kind": "neighbor_search"},
                    )
                )

        cypher_data = await self._load_json_response(
            CYPHER_GENERATION_PROMPT,
            f"问题: {question}\n实体: {analysis.entities}",
            fallback={"queries": []},
        )
        raw_cypher_queries = cypher_data.get("queries", [])
        if not isinstance(raw_cypher_queries, list):
            raw_cypher_queries = []
        for cypher in raw_cypher_queries[:2]:
            if not isinstance(cypher, str) or not self._is_read_only_cypher(cypher):
                continue
            try:
                records = await self.knowledge_graph.execute_cypher(cypher)
            except Exception:
                continue
            for record in records[:per_query_limit]:
                contexts.append(
                    RetrievedContext(
                        content=str(record),
                        source="knowledge_graph",
                        score=0.82,
                        retrieval_type="graph",
                        metadata={"query": question, "kind": "cypher", "cypher": cypher},
                    )
                )

        return contexts

    async def _web_retrieve(
        self,
        analysis: QueryAnalysis,
        top_k: int,
    ) -> list[RetrievedContext]:
        if not self.web_search or not getattr(self.web_search, "enabled", False):
            return []

        queries = analysis.queries[:2] or [analysis.question]
        contexts: list[RetrievedContext] = []
        search_batches = await asyncio.gather(
            *(self.web_search.search(query, top_k=top_k) for query in queries),
            return_exceptions=True,
        )
        items_with_queries: list[tuple[Any, str]] = []
        for query, batch in zip(queries, search_batches, strict=False):
            if isinstance(batch, Exception):
                continue
            items_with_queries.extend((item, query) for item in batch)

        page_batches = await asyncio.gather(
            *(self._read_web_page(item, query) for item, query in items_with_queries),
            return_exceptions=True,
        )
        for (item, query), page_batch in zip(items_with_queries, page_batches, strict=False):
            if not getattr(item, "title", "") and not getattr(item, "snippet", ""):
                continue
            page_contexts = [] if isinstance(page_batch, Exception) else page_batch
            if page_contexts:
                contexts.extend(page_contexts)
                continue

            contexts.append(
                RetrievedContext(
                    content=f"{item.title}\n{item.snippet}".strip(),
                    source=item.url or item.provider,
                    score=coerce_float(item.score),
                    retrieval_type="web",
                    metadata={
                        "query": query,
                        "provider": item.provider,
                        "title": item.title,
                        "url": item.url,
                    },
                )
            )
        return contexts

    async def _read_web_page(self, item: Any, query: str) -> list[RetrievedContext]:
        if not self.web_page_reader or not getattr(self.web_page_reader, "enabled", False):
            return []
        try:
            chunks = await self.web_page_reader.read(
                url=item.url,
                title=item.title,
                score=coerce_float(item.score),
            )
        except Exception:
            return []

        contexts: list[RetrievedContext] = []
        for chunk in chunks:
            contexts.append(
                RetrievedContext(
                    content=chunk.content,
                    source=chunk.url,
                    score=coerce_float(item.score),
                    retrieval_type="web_page",
                    metadata={
                        "query": query,
                        "provider": item.provider,
                        "title": chunk.title,
                        "url": chunk.url,
                        "chunk_index": chunk.chunk_index,
                        "retrieved_at": chunk.retrieved_at,
                        **chunk.metadata,
                    },
                )
            )
        return contexts

    async def _load_json_response(
        self,
        system_prompt: str,
        user_content: str,
        fallback: dict[str, Any],
    ) -> dict[str, Any]:
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_content),
        ]
        try:
            resp = await self.llm.ainvoke(messages)
        except Exception:
            return fallback

        return parse_json_object(resp.content) or fallback

    @staticmethod
    def _is_read_only_cypher(cypher: str) -> bool:
        """Allow model-generated read queries while rejecting mutating Cypher."""
        cleaned = re.sub(r"//.*?$|/\*.*?\*/", " ", cypher, flags=re.MULTILINE | re.DOTALL).strip()
        if not re.match(r"^(MATCH|OPTIONAL\s+MATCH)\b", cleaned, flags=re.IGNORECASE):
            return False
        forbidden = r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|CALL|LOAD\s+CSV|FOREACH)\b"
        return re.search(forbidden, cleaned, flags=re.IGNORECASE) is None

    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        weight_map = {"vector": 1.0, "graph": 1.2, "web": 1.05, "web_page": 1.15, "hybrid": 1.1}
        deduped: dict[str, RetrievedContext] = {}

        for ctx in contexts:
            content = " ".join(str(ctx.content or "").split()).strip()
            if not content:
                continue
            source = str(ctx.source or ctx.metadata.get("url") or "unknown").strip()
            weighted_score = coerce_float(ctx.score) * weight_map.get(ctx.retrieval_type, 1.0)
            ranked = RetrievedContext(
                content=content,
                source=source,
                score=min(max(weighted_score, 0.0), 1.0),
                retrieval_type=ctx.retrieval_type,
                metadata=dict(ctx.metadata),
            )
            canonical_url = str(ranked.metadata.get("url", "")).split("#", 1)[0].rstrip("/")
            identity = f"{canonical_url or source}\0{content.casefold()}"
            key = hashlib.sha256(identity.encode("utf-8")).hexdigest()
            existing = deduped.get(key)
            if existing is None or ranked.score > existing.score:
                deduped[key] = ranked

        ordered = list(deduped.values())
        ordered.sort(key=lambda item: item.score, reverse=True)
        return ordered
