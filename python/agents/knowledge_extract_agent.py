"""
知识抽取 Agent — 从文档块中提取实体、关系、事件，构建知识图谱三元组

核心能力:
  1. 命名实体识别 (NER)
  2. 关系抽取 (RE)
  3. 事件抽取
  4. 三元组生成 → 写入 Neo4j
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agents.doc_parser_agent import DocumentChunk
from utils.json_utils import coerce_float, parse_json_object
from utils.openai_clients import create_chat_model

EXTRACTION_SYSTEM_PROMPT = """\
你是一个专业的知识抽取引擎。给定一段文本，请提取其中的：
1. **实体 (entities)**：人名、组织、地点、产品、技术、概念等
2. **关系 (relations)**：实体之间的关系，用三元组 (头实体, 关系, 尾实体) 表示
3. **事件 (events)**：文本中提到的事件，包含触发词和参与者

请严格按照以下 JSON 格式返回：
{
  "entities": [
    {"name": "实体名", "type": "实体类型", "description": "简短描述"}
  ],
  "relations": [
    {"head": "头实体", "relation": "关系类型", "tail": "尾实体", "confidence": 0.95}
  ],
  "events": [
    {"trigger": "触发词", "type": "事件类型", "participants": ["参与者1"]}
  ]
}

注意:
- 实体类型包括: Person, Organization, Location, Product, Technology, Concept, Event, Time
- 关系类型包括: belongs_to, works_at, located_in, developed_by, related_to, part_of, uses, depends_on
- confidence 为 0-1 之间的浮点数
- 只返回 JSON，不要包含其他文字
"""


@dataclass
class Entity:
    name: str
    type: str
    description: str = ""
    properties: dict[str, Any] = field(default_factory=dict)

    @property
    def node_label(self) -> str:
        return self.type.replace(" ", "_")


@dataclass
class Relation:
    head: str
    relation: str
    tail: str
    confidence: float = 0.0
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEvent:
    trigger: str
    type: str
    participants: list[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    entities: list[Entity]
    relations: list[Relation]
    events: list[KnowledgeEvent]
    source_chunk_id: str = ""


class KnowledgeExtractAgent:
    """
    知识抽取 Agent

    工作流:
      receive_chunks → extract_per_chunk → deduplicate → resolve_entities → output_triples
    """

    BATCH_SIZE = 5

    def __init__(self, llm: Any = None) -> None:
        self.llm = llm or create_chat_model()

    # ── public API ───────────────────────────────────────────

    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        """从一组文档块中抽取知识"""
        results: list[ExtractionResult] = []
        for i in range(0, len(chunks), self.BATCH_SIZE):
            batch = chunks[i : i + self.BATCH_SIZE]
            for chunk in batch:
                result = await self._extract_from_chunk(chunk)
                results.append(result)
        merged = self._deduplicate(results)
        return merged

    async def extract_single(self, text: str, chunk_id: str = "") -> ExtractionResult:
        """从单段文本中抽取知识"""
        return await self._extract_from_text(text, chunk_id)

    # ── core extraction ──────────────────────────────────────

    async def _extract_from_chunk(self, chunk: DocumentChunk) -> ExtractionResult:
        return await self._extract_from_text(chunk.content, chunk.chunk_id)

    async def _extract_from_text(self, text: str, source_id: str) -> ExtractionResult:
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=f"请从以下文本中抽取知识：\n\n{text}"),
        ]
        try:
            resp = await self.llm.ainvoke(messages)
        except Exception:
            return ExtractionResult(entities=[], relations=[], events=[], source_chunk_id=source_id)
        return self._parse_response(str(resp.content), source_id)

    def _parse_response(self, raw: str, source_id: str) -> ExtractionResult:
        data = parse_json_object(raw)
        if data is None:
            return ExtractionResult(entities=[], relations=[], events=[], source_chunk_id=source_id)

        raw_entities = data.get("entities", [])
        raw_relations = data.get("relations", [])
        raw_events = data.get("events", [])
        raw_entities = raw_entities if isinstance(raw_entities, list) else []
        raw_relations = raw_relations if isinstance(raw_relations, list) else []
        raw_events = raw_events if isinstance(raw_events, list) else []

        entities = [
            Entity(
                name=e.get("name", ""),
                type=e.get("type", "Concept"),
                description=e.get("description", ""),
            )
            for e in raw_entities
            if isinstance(e, dict) and e.get("name")
        ]
        relations = [
            Relation(
                head=r.get("head", ""),
                relation=r.get("relation", "related_to"),
                tail=r.get("tail", ""),
                confidence=min(max(coerce_float(r.get("confidence"), 0.5), 0.0), 1.0),
            )
            for r in raw_relations
            if isinstance(r, dict) and r.get("head") and r.get("tail")
        ]
        events = [
            KnowledgeEvent(
                trigger=ev.get("trigger", ""),
                type=ev.get("type", ""),
                participants=ev.get("participants", []),
            )
            for ev in raw_events
            if isinstance(ev, dict)
        ]
        return ExtractionResult(
            entities=entities,
            relations=relations,
            events=events,
            source_chunk_id=source_id,
        )

    # ── deduplication & entity resolution ────────────────────

    @staticmethod
    def _deduplicate(results: list[ExtractionResult]) -> list[ExtractionResult]:
        """
        跨 chunk 去重: 同名同类型实体合并，关系去重
        """
        seen_entities: dict[str, Entity] = {}
        seen_relations: set[tuple[str, str, str]] = set()
        deduped: list[ExtractionResult] = []

        for result in results:
            unique_entities: list[Entity] = []
            for ent in result.entities:
                key = f"{ent.name}::{ent.type}"
                if key not in seen_entities:
                    seen_entities[key] = ent
                    unique_entities.append(ent)

            unique_relations: list[Relation] = []
            for rel in result.relations:
                key = (rel.head, rel.relation, rel.tail)
                if key not in seen_relations:
                    seen_relations.add(key)
                    unique_relations.append(rel)

            deduped.append(
                ExtractionResult(
                    entities=unique_entities,
                    relations=unique_relations,
                    events=result.events,
                    source_chunk_id=result.source_chunk_id,
                )
            )
        return deduped
