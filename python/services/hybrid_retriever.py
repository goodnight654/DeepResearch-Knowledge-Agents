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

import json
from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings


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
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_client_base_url,
            temperature=0,
        )

    async def analyze(self, question: str) -> QueryAnalysis:
        data = await self._load_json_response(
            QUERY_REWRITE_PROMPT,
            question,
            fallback={"queries": [question], "entities": [], "keywords": []},
        )
        queries = [q.strip() for q in data.get("queries", []) if isinstance(q, str) and q.strip()]
        if question not in queries:
            queries.insert(0, question)
        return QueryAnalysis(
            question=question,
            queries=queries[:3],
            entities=[e for e in data.get("entities", []) if isinstance(e, str) and e.strip()][:6],
            keywords=[k for k in data.get("keywords", []) if isinstance(k, str) and k.strip()][:10],
        )

    async def retrieve(
        self,
        question: str,
        top_k: int = 8,
        per_query_limit: int = 5,
        web_limit: int | None = None,
    ) -> tuple[QueryAnalysis, list[RetrievedContext]]:
        analysis = await self.analyze(question)
        vector_contexts = await self._vector_retrieve(analysis, per_query_limit)
        graph_contexts = await self._graph_retrieve(question, analysis, per_query_limit)
        web_contexts = await self._web_retrieve(analysis, web_limit or min(top_k, settings.web_search_top_k))
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
            results = await self.vector_store.search(query, top_k=per_query_limit)
            for doc, score in results:
                contexts.append(
                    RetrievedContext(
                        content=doc.get("content", ""),
                        source=doc.get("source", "vector_store"),
                        score=score,
                        retrieval_type="vector",
                        metadata={
                            **doc.get("metadata", {}),
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
            entity_hits = await self.knowledge_graph.search_entities(entity, limit=3)
            for hit in entity_hits:
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

            neighbors = await self.knowledge_graph.get_neighbors(entity, hops=2)
            for record in neighbors[:per_query_limit]:
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
        for cypher in cypher_data.get("queries", [])[:2]:
            if not isinstance(cypher, str) or not cypher.strip():
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
        for query in queries:
            try:
                results = await self.web_search.search(query, top_k=top_k)
            except Exception:
                continue
            for item in results:
                page_contexts = await self._read_web_page(item, query)
                if page_contexts:
                    contexts.extend(page_contexts)
                    continue

                contexts.append(
                    RetrievedContext(
                        content=f"{item.title}\n{item.snippet}",
                        source=item.url or item.provider,
                        score=float(item.score),
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
                score=float(item.score),
            )
        except Exception:
            return []

        contexts: list[RetrievedContext] = []
        for chunk in chunks:
            contexts.append(
                RetrievedContext(
                    content=chunk.content,
                    source=chunk.url,
                    score=float(item.score),
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

        cleaned = str(resp.content).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return fallback
        return data if isinstance(data, dict) else fallback

    @staticmethod
    def _hybrid_rerank(contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        weight_map = {"vector": 1.0, "graph": 1.2, "web": 1.05, "web_page": 1.15, "hybrid": 1.1}
        deduped: dict[tuple[str, str], RetrievedContext] = {}

        for ctx in contexts:
            ctx.score *= weight_map.get(ctx.retrieval_type, 1.0)
            key = (ctx.source, ctx.content[:160])
            existing = deduped.get(key)
            if existing is None or ctx.score > existing.score:
                deduped[key] = ctx

        ordered = list(deduped.values())
        ordered.sort(key=lambda item: item.score, reverse=True)
        return ordered
