import pytest
from openpyxl import Workbook

from agents.doc_parser_agent import DocParserAgent, DocType
from services.graph_rag import GraphRAGContext, GraphRAGPipeline
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService


class FailingLLM:
    async def ainvoke(self, messages):
        raise RuntimeError("offline")


async def test_doc_parser_reads_text_and_rejects_unknown_files(tmp_path):
    text_path = tmp_path / "sample.txt"
    text_path.write_text("hello world", encoding="utf-8")
    parser = DocParserAgent(llm=FailingLLM())
    chunks = await parser.parse(str(text_path))
    assert chunks[0].content == "hello world"
    assert chunks[0].doc_type == DocType.TEXT

    unknown = tmp_path / "sample.bin"
    unknown.write_bytes(b"binary")
    with pytest.raises(ValueError, match="unsupported document type"):
        await parser.parse(str(unknown))


async def test_doc_parser_reads_excel(tmp_path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.append(["name", "value"])
    sheet.append(["alpha", 42])
    path = tmp_path / "sample.xlsx"
    workbook.save(path)

    chunks = await DocParserAgent(llm=FailingLLM()).parse(str(path))
    assert chunks[0].doc_type == DocType.TABLE
    assert "name: alpha" in chunks[0].content
    assert "value: 42" in chunks[0].content


async def test_uninitialized_vector_store_degrades_reads_and_reports_status():
    store = VectorStoreService()
    assert await store.search("query") == []
    stats = await store.get_stats()
    assert stats["available"] is False
    assert stats["backend"] in {"chroma", "pgvector"}


async def test_uninitialized_knowledge_graph_degrades_reads_and_reports_status():
    graph = KnowledgeGraphService()
    assert await graph.execute_cypher("MATCH (n) RETURN n") == []
    assert (await graph.get_stats())["available"] is False


def test_relation_type_is_safe_for_cypher_identifier():
    assert KnowledgeGraphService._sanitize_relation_type("works at") == "WORKS_AT"
    assert KnowledgeGraphService._sanitize_relation_type("` DELETE n //") == "DELETE_N"
    assert KnowledgeGraphService._sanitize_relation_type("123") == "REL_123"


def test_graphrag_rerank_does_not_mutate_inputs():
    original = GraphRAGContext("path", "path", 0.9)
    ranked = GraphRAGPipeline._cross_rerank([original], "query")
    assert original.score == 0.9
    assert ranked[0].score == 1.0
