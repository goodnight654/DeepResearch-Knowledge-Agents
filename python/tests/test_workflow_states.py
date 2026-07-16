from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from agents.knowledge_update_agent import ChangeType, DocumentChange, UpdateResult
from agents.qa_agent import QAResult, QueryIntent
from orchestrator.graph import _build_ingest_graph, _build_qa_graph, _build_update_graph


class FakeParser:
    async def parse_batch(self, paths):
        return [DocumentChunk("content", "doc", 0, DocType.TEXT)]


class FakeExtractor:
    async def extract(self, chunks):
        return [
            ExtractionResult(
                entities=[Entity("Alice", "Person", "")],
                relations=[Relation("Alice", "works_at", "Example", 0.9)],
                events=[],
                source_chunk_id=chunks[0].chunk_id,
            )
        ]


class FakeVectorStore:
    async def add_chunks(self, chunks):
        return len(chunks)


class FakeKnowledgeGraph:
    def __init__(self):
        self.entities = 0
        self.relations = 0

    async def upsert_entity(self, entity, **kwargs):
        self.entities += 1

    async def add_relation(self, relation, **kwargs):
        self.relations += 1


async def test_ingest_typed_state_merges_parallel_store_results():
    knowledge_graph = FakeKnowledgeGraph()
    workflow = _build_ingest_graph(FakeParser(), FakeExtractor(), FakeVectorStore(), knowledge_graph)
    state = await workflow.ainvoke({"file_paths": ["doc.txt"]})
    assert len(state["chunks"]) == 1
    assert len(state["extractions"]) == 1
    assert state["vectors_stored"] == 1
    assert state["entities_stored"] == 1
    assert knowledge_graph.relations == 1


class FakeQAAgent:
    async def answer(self, question):
        return QAResult(question, "answer", [], QueryIntent.FACTOID, 0.5)


async def test_qa_typed_state_preserves_question():
    state = await _build_qa_graph(FakeQAAgent()).ainvoke({"question": "question"})
    assert state["question"] == "question"
    assert state["result"].answer == "answer"


class RetryUpdateAgent:
    def __init__(self):
        self.calls = 0

    async def process_batch(self, changes):
        self.calls += 1
        return [UpdateResult(change=changes[0], success=self.calls > 1)]


async def test_update_typed_state_retries_once_and_preserves_change():
    agent = RetryUpdateAgent()
    change = DocumentChange("doc.txt", ChangeType.MODIFIED)
    state = await _build_update_graph(agent).ainvoke({"changes": [change]})
    assert agent.calls == 2
    assert state["results"][0].success is True
    assert state["results"][0].change is change
