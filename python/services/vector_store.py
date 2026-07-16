"""
向量存储服务 — 支持 ChromaDB / PGVector 双后端

职责:
  1. 文档块向量化 (Embedding)
  2. 向量存储 & 检索
  3. 按 doc_id 删除（支持增量更新）
"""

from __future__ import annotations

import asyncio
from typing import Any

from agents.doc_parser_agent import DocumentChunk
from config import settings
from utils.openai_clients import create_embeddings


class VectorStoreService:
    """向量库统一接口，底层可切换 ChromaDB / PGVector"""

    COLLECTION_NAME = "knowledge_chunks"

    def __init__(self) -> None:
        self.embeddings = create_embeddings()
        self._store: Any = None
        self._backend = settings.vector_store_type
        self._initialization_error = ""

    @property
    def available(self) -> bool:
        return self._store is not None

    # ── initialization ───────────────────────────────────────

    async def init(self) -> None:
        try:
            if self._backend == "chroma":
                await self._init_chroma()
            else:
                await self._init_pgvector()
            self._initialization_error = ""
        except Exception as exc:
            self._store = None
            self._initialization_error = str(exc)
            raise

    async def _init_chroma(self) -> None:
        import chromadb

        def initialize() -> Any:
            client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
            return client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )

        self._store = await asyncio.to_thread(initialize)

    async def _init_pgvector(self) -> None:
        from langchain_community.vectorstores import PGVector

        self._store = await asyncio.to_thread(
            PGVector,
            connection_string=settings.pgvector_dsn,
            collection_name=self.COLLECTION_NAME,
            embedding_function=self.embeddings,
        )

    # ── CRUD ─────────────────────────────────────────────────

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """向量化并存储文档块"""
        if not chunks:
            return 0
        if not self.available:
            raise RuntimeError("vector store is not initialized")

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {
                "doc_id": c.doc_id,
                "doc_type": c.doc_type.value,
                "source": c.metadata.get("source", ""),
                "chunk_index": c.chunk_index,
            }
            for c in chunks
        ]

        if self._backend == "chroma":
            vectors = await self.embeddings.aembed_documents(texts)
            self._store.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)
        else:
            await self._store.aadd_texts(texts=texts, metadatas=metadatas, ids=ids)

        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """语义搜索，返回 (文档, 分数) 列表"""
        if not self.available or not query.strip():
            return []
        if self._backend == "chroma":
            q_vec = await self.embeddings.aembed_query(query)
            results = self._store.query(
                query_embeddings=[q_vec], n_results=top_k, include=["documents", "metadatas", "distances"]
            )
            out: list[tuple[dict, float]] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists, strict=False):
                score = min(max(1.0 - float(dist), 0.0), 1.0)  # cosine distance → similarity
                out.append(({"content": doc, "source": meta.get("source", ""), "metadata": meta}, score))
            return out
        else:
            if hasattr(self._store, "asimilarity_search_with_relevance_scores"):
                results = await self._store.asimilarity_search_with_relevance_scores(query, k=top_k)

                def normalize(value: float) -> float:
                    return min(max(float(value), 0.0), 1.0)
            else:
                results = await self._store.asimilarity_search_with_score(query, k=top_k)

                def normalize(value: float) -> float:
                    return 1.0 / (1.0 + max(float(value), 0.0))

            return [
                (
                    {
                        "content": doc.page_content,
                        "source": doc.metadata.get("source", ""),
                        "metadata": doc.metadata,
                    },
                    normalize(score),
                )
                for doc, score in results
            ]

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除所有相关向量"""
        if not self.available:
            return 0
        if self._backend == "chroma":
            existing = self._store.get(where={"doc_id": doc_id}, include=[])
            ids = existing.get("ids", [])
            if ids:
                self._store.delete(ids=ids)
            return len(ids)
        return 0

    async def get_stats(self) -> dict:
        """获取向量库统计信息"""
        if not self.available:
            return {
                "backend": self._backend,
                "available": False,
                "collection": self.COLLECTION_NAME,
                "error": self._initialization_error,
            }
        if self._backend == "chroma":
            count = self._store.count()
            return {
                "backend": "chroma",
                "available": True,
                "total_vectors": count,
                "collection": self.COLLECTION_NAME,
            }
        return {"backend": "pgvector", "available": True, "collection": self.COLLECTION_NAME}
