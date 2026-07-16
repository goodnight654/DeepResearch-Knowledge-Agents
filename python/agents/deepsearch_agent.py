"""
DeepResearch Agent — 计划驱动的多轮检索、网页阅读、证据归因与综合回答

目标:
  1. 将复杂问题拆成可执行的搜索计划
  2. 针对每轮发现的知识缺口继续追问和检索
  3. 最终输出带有研究轨迹的综合答案

说明:
  - `research()` / `search()` 保留一体化调用能力
  - 同时暴露 step-level API，供 LangGraph 多节点编排使用
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from config import settings
from services.hybrid_retriever import HybridRetriever, RetrievedContext
from utils.json_utils import coerce_float, parse_json_object
from utils.openai_clients import create_chat_model

logger = logging.getLogger(__name__)


@dataclass
class DeepResearchStep:
    iteration: int
    focus: str
    queries: list[str]
    summary: str
    gaps: list[str] = field(default_factory=list)
    contexts: list[RetrievedContext] = field(default_factory=list)


@dataclass
class DeepResearchEvidence:
    evidence_id: str
    source: str
    title: str
    quote: str
    confidence: float
    retrieved_at: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DeepResearchResult:
    question: str
    answer: str
    executive_summary: str
    contexts: list[RetrievedContext]
    steps: list[DeepResearchStep]
    evidence: list[DeepResearchEvidence]
    confidence: float
    cited_evidence_ids: list[str] = field(default_factory=list)
    citation_warnings: list[str] = field(default_factory=list)

    @property
    def iterations(self) -> int:
        return len(self.steps)


SEARCH_PLAN_PROMPT = """\
你是一个 DeepResearch 研究规划器。请把用户问题拆解成一个搜索计划。
返回 JSON，格式如下：
{
  "goal": "本次搜索的核心目标",
  "sub_questions": ["子问题1", "子问题2"],
  "initial_queries": ["第一轮检索查询1", "第一轮检索查询2"],
  "expected_evidence": ["希望找到的证据类型1"]
}
只返回 JSON。
"""

ROUND_SUMMARY_PROMPT = """\
你是一个研究分析助手。请根据当前轮次检索到的材料，总结本轮已经获得的关键信息。
要求：
1. 聚焦当前搜索焦点
2. 只总结有依据的信息
3. 明确指出仍不清楚的部分
"""

GAP_ANALYSIS_PROMPT = """\
你是一个 DeepResearch gap 分析器。请判断当前材料是否足以回答原问题。
返回 JSON，格式如下：
{
  "answered": true,
  "confidence": 0.82,
  "next_focus": "下一轮应该继续搜索的焦点",
  "gaps": ["仍缺少的信息1"],
  "follow_up_queries": ["下一轮查询1", "下一轮查询2"]
}
只返回 JSON。
"""

FINAL_SYNTHESIS_PROMPT = """\
你是一个高级研究型问答助手。请根据搜索步骤和上下文，生成最终回答。
要求：
1. 所有关键结论必须引用证据编号，例如 [E1]、[E2]
2. 不要引用没有出现在证据列表中的编号
3. 如果证据不足，明确说明哪些结论无法确认
返回 JSON，格式如下：
{
  "executive_summary": "一段 2-4 句的结论摘要",
  "answer": "完整回答，关键结论使用 [E1] 这类证据编号引用",
  "confidence": 0.85
}
只返回 JSON。
"""


class DeepResearchAgent:
    """多轮 DeepResearch Agent。"""

    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
        web_search: Any = None,
        web_page_reader: Any = None,
        retriever: HybridRetriever | None = None,
        llm: Any = None,
    ) -> None:
        self.llm = llm or create_chat_model()
        self.retriever = retriever or HybridRetriever(
            vector_store=vector_store,
            knowledge_graph=knowledge_graph,
            web_search=web_search,
            web_page_reader=web_page_reader,
            llm=self.llm,
        )
        self.max_iterations = settings.deepresearch_max_iterations
        self.queries_per_iteration = settings.deepresearch_queries_per_iteration
        self.contexts_per_query = settings.deepresearch_contexts_per_query
        self.final_context_limit = settings.deepresearch_final_context_limit

    async def research(self, question: str) -> DeepResearchResult:
        question = self.normalize_question(question)
        plan = await self.plan_search(question)
        focus = str(plan.get("goal") or question)
        frontier = self.initial_frontier(plan, question)

        all_contexts: list[RetrievedContext] = []
        steps: list[DeepResearchStep] = []
        attempted_queries: set[str] = set()
        confidence = 0.0

        for iteration in range(1, self.max_iterations + 1):
            queries = self.select_queries(frontier, attempted_queries, fallback=question)
            if not queries:
                break
            attempted_queries.update(query.casefold() for query in queries)
            round_contexts = await self.retrieve_round(queries)
            all_contexts = self.merge_contexts(all_contexts + round_contexts)

            summary = await self.summarize_round(question, focus, queries, all_contexts)
            gap_state = await self.analyze_gaps(question, focus, steps, summary, all_contexts)
            confidence = coerce_float(gap_state.get("confidence"), confidence or 0.0)

            steps.append(
                self.make_step(
                    iteration=iteration,
                    focus=focus,
                    queries=queries,
                    summary=summary,
                    gaps=self.clean_list(gap_state.get("gaps")),
                    contexts=round_contexts,
                )
            )

            follow_up = self.select_queries(
                self.clean_list(gap_state.get("follow_up_queries")),
                attempted_queries,
            )
            if self.should_finalize(iteration, gap_state, follow_up):
                break

            focus = str(gap_state.get("next_focus") or focus)
            frontier = follow_up

        contexts = self.select_final_contexts(all_contexts)
        evidence = self.build_evidence(contexts)
        executive_summary, answer, final_confidence, cited_ids, citation_warnings = await self.compose_answer(
            question=question,
            plan=plan,
            steps=steps,
            contexts=contexts,
            evidence=evidence,
            fallback_confidence=confidence,
        )
        return self.build_result(
            question=question,
            answer=answer,
            executive_summary=executive_summary,
            contexts=contexts,
            steps=steps,
            evidence=evidence,
            confidence=final_confidence,
            cited_evidence_ids=cited_ids,
            citation_warnings=citation_warnings,
        )

    async def search(self, question: str) -> DeepResearchResult:
        return await self.research(question)

    async def plan_search(self, question: str) -> dict[str, Any]:
        question = self.normalize_question(question)
        plan = await self._load_json_response(
            SEARCH_PLAN_PROMPT,
            question,
            fallback={
                "goal": question,
                "sub_questions": [question],
                "initial_queries": [question],
                "expected_evidence": [],
            },
        )
        plan["sub_questions"] = self.clean_list(plan.get("sub_questions"))
        plan["initial_queries"] = self.clean_list(plan.get("initial_queries"))
        plan["expected_evidence"] = self.clean_list(plan.get("expected_evidence"))
        return plan

    def initial_frontier(self, plan: dict[str, Any], question: str) -> list[str]:
        frontier = self.clean_list(plan.get("initial_queries")) or self.clean_list(plan.get("sub_questions"))
        return frontier or [question]

    async def retrieve_round(self, queries: list[str]) -> list[RetrievedContext]:
        contexts: list[RetrievedContext] = []
        selected = queries[: self.queries_per_iteration]
        results = await asyncio.gather(
            *(
                self.retriever.retrieve(
                    query,
                    top_k=self.contexts_per_query,
                    per_query_limit=self.contexts_per_query,
                    web_limit=min(self.contexts_per_query, settings.web_search_top_k),
                )
                for query in selected
            ),
            return_exceptions=True,
        )
        for query, result in zip(selected, results, strict=False):
            if isinstance(result, Exception):
                logger.warning("DeepResearch query failed for %r: %s", query, result)
                continue
            _, hits = result
            contexts.extend(hits)
        return contexts

    async def summarize_round(
        self,
        question: str,
        focus: str,
        queries: list[str],
        contexts: list[RetrievedContext],
    ) -> str:
        context_text = self.format_contexts(contexts[: self.final_context_limit])
        messages = [
            SystemMessage(content=ROUND_SUMMARY_PROMPT),
            HumanMessage(
                content=(
                    f"原问题: {question}\n"
                    f"当前焦点: {focus}\n"
                    f"本轮查询: {queries}\n\n"
                    f"当前已收集材料:\n{context_text}"
                )
            ),
        ]
        try:
            resp = await self.llm.ainvoke(messages)
            return str(resp.content)
        except Exception:
            return f"围绕“{focus}”完成了 {len(queries)} 个查询，当前累计检索到 {len(contexts)} 条上下文。"

    async def analyze_gaps(
        self,
        question: str,
        focus: str,
        steps: list[DeepResearchStep],
        summary: str,
        contexts: list[RetrievedContext],
    ) -> dict[str, Any]:
        prior_summaries = "\n\n".join(
            f"第 {step.iteration} 轮 / 焦点: {step.focus}\n总结: {step.summary}" for step in steps[-2:]
        )
        payload = (
            f"原问题: {question}\n"
            f"当前焦点: {focus}\n"
            f"已有阶段总结:\n{prior_summaries or '[无]'}\n\n"
            f"本轮总结:\n{summary}\n\n"
            f"当前可用上下文数: {len(contexts)}\n"
            f"当前独立来源数: {len(self.context_sources(contexts))}\n"
            f"建议最低证据数: {settings.deepresearch_min_evidence}\n"
            f"建议最低独立来源数: {settings.deepresearch_min_sources}"
        )
        enough_evidence = len(contexts) >= settings.deepresearch_min_evidence
        enough_sources = len(self.context_sources(contexts)) >= settings.deepresearch_min_sources
        result = await self._load_json_response(
            GAP_ANALYSIS_PROMPT,
            payload,
            fallback={
                "answered": enough_evidence and enough_sources,
                "confidence": 0.0,
                "next_focus": focus,
                "gaps": [],
                "follow_up_queries": [],
            },
        )
        result["gaps"] = self.clean_list(result.get("gaps"))
        result["confidence"] = min(max(coerce_float(result.get("confidence")), 0.0), 1.0)
        result["answered"] = bool(result.get("answered")) and enough_evidence and enough_sources

        follow_up = self.clean_list(result.get("follow_up_queries"))
        if not result["answered"] and not follow_up and len(steps) + 1 < self.max_iterations:
            variants = (
                f"{question} 官方资料 原始来源",
                f"{question} 独立分析 交叉验证",
                f"{question} 最新证据 风险限制",
            )
            follow_up = [variants[min(len(steps), len(variants) - 1)]]
            if not enough_evidence:
                result["gaps"] = self.clean_list(result["gaps"] + ["证据数量不足"])
            if not enough_sources:
                result["gaps"] = self.clean_list(result["gaps"] + ["独立来源不足"])
        result["follow_up_queries"] = follow_up
        return result

    async def compose_answer(
        self,
        question: str,
        plan: dict[str, Any],
        steps: list[DeepResearchStep],
        contexts: list[RetrievedContext],
        evidence: list[DeepResearchEvidence],
        fallback_confidence: float,
    ) -> tuple[str, str, float, list[str], list[str]]:
        steps_text = "\n\n".join(
            (
                f"第 {step.iteration} 轮\n"
                f"焦点: {step.focus}\n"
                f"查询: {step.queries}\n"
                f"总结: {step.summary}\n"
                f"缺口: {step.gaps or ['无']}"
            )
            for step in steps
        )
        payload = (
            f"原问题: {question}\n"
            f"搜索目标: {plan.get('goal', question)}\n"
            f"预期证据: {plan.get('expected_evidence', [])}\n\n"
            f"搜索过程:\n{steps_text}\n\n"
            f"证据列表:\n{self.format_evidence(evidence)}\n\n"
            f"关键上下文:\n{self.format_contexts(contexts)}"
        )
        fallback_answer = self._fallback_answer(evidence, steps)
        result = await self._load_json_response(
            FINAL_SYNTHESIS_PROMPT,
            payload,
            fallback={
                "executive_summary": steps[-1].summary if steps else "暂无搜索摘要。",
                "answer": fallback_answer,
                "confidence": fallback_confidence,
            },
        )
        executive_summary = str(result.get("executive_summary") or "暂无搜索摘要。")
        answer = str(result.get("answer") or executive_summary)
        answer, cited_ids, warnings = self.validate_citations(answer, evidence)
        model_confidence = coerce_float(result.get("confidence"), fallback_confidence or 0.0)
        confidence = self.calibrate_confidence(model_confidence, evidence, cited_ids)
        return executive_summary, answer, confidence, cited_ids, warnings

    @staticmethod
    def build_result(
        question: str,
        answer: str,
        executive_summary: str,
        contexts: list[RetrievedContext],
        steps: list[DeepResearchStep],
        evidence: list[DeepResearchEvidence],
        confidence: float,
        cited_evidence_ids: list[str] | None = None,
        citation_warnings: list[str] | None = None,
    ) -> DeepResearchResult:
        return DeepResearchResult(
            question=question,
            answer=answer,
            executive_summary=executive_summary,
            contexts=contexts,
            steps=steps,
            evidence=evidence,
            confidence=min(max(coerce_float(confidence), 0.0), 1.0),
            cited_evidence_ids=cited_evidence_ids or [],
            citation_warnings=citation_warnings or [],
        )

    def make_step(
        self,
        iteration: int,
        focus: str,
        queries: list[str],
        summary: str,
        gaps: list[str],
        contexts: list[RetrievedContext],
    ) -> DeepResearchStep:
        return DeepResearchStep(
            iteration=iteration,
            focus=focus,
            queries=queries,
            summary=summary,
            gaps=gaps,
            contexts=contexts[: self.final_context_limit],
        )

    def should_finalize(
        self,
        iteration: int,
        gap_state: dict[str, Any],
        follow_up_queries: list[str],
    ) -> bool:
        if bool(gap_state.get("answered")):
            return True
        if iteration >= self.max_iterations:
            return True
        return not bool(follow_up_queries)

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
    def normalize_question(question: str) -> str:
        value = " ".join(str(question or "").split()).strip()
        if not value:
            raise ValueError("question must not be empty")
        return value

    @staticmethod
    def clean_list(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        values = [str(item).strip() for item in raw if item is not None and str(item).strip()]
        return list(dict.fromkeys(values))

    def select_queries(
        self,
        candidates: Any,
        attempted_queries: set[str],
        fallback: str = "",
    ) -> list[str]:
        values = self.clean_list(candidates)
        if not values and fallback:
            values = [fallback]
        selected = [value for value in values if value.casefold() not in attempted_queries]
        return selected[: self.queries_per_iteration]

    def merge_contexts(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        deduped: dict[tuple[str, str], RetrievedContext] = {}
        for ctx in contexts:
            content = " ".join(str(ctx.content or "").split()).strip()
            if not content:
                continue
            source = str(ctx.source or ctx.metadata.get("url") or "unknown").strip()
            candidate = RetrievedContext(
                content=content,
                source=source,
                score=min(max(coerce_float(ctx.score), 0.0), 1.0),
                retrieval_type=ctx.retrieval_type,
                metadata=dict(ctx.metadata),
            )
            key = (self.context_source(candidate), content.casefold())
            existing = deduped.get(key)
            if existing is None or candidate.score > existing.score:
                deduped[key] = candidate
        merged = list(deduped.values())
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged

    def select_final_contexts(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        """Prefer source diversity before filling remaining slots by score."""
        ordered = self.merge_contexts(contexts)
        selected: list[RetrievedContext] = []
        seen_sources: set[str] = set()
        for ctx in ordered:
            source_key = self.context_source(ctx)
            if source_key in seen_sources:
                continue
            selected.append(ctx)
            seen_sources.add(source_key)
            if len(selected) >= self.final_context_limit:
                return selected
        for ctx in ordered:
            if ctx not in selected:
                selected.append(ctx)
                if len(selected) >= self.final_context_limit:
                    break
        return selected

    @staticmethod
    def context_source(context: RetrievedContext) -> str:
        return (
            str(context.metadata.get("url") or context.source or "unknown")
            .split("#", 1)[0]
            .rstrip("/")
            .casefold()
        )

    @classmethod
    def context_sources(cls, contexts: list[RetrievedContext]) -> set[str]:
        return {cls.context_source(context) for context in contexts}

    @staticmethod
    def format_contexts(contexts: list[RetrievedContext]) -> str:
        if not contexts:
            return "[无可用上下文]"
        return "\n\n".join(
            f"[来源 {index + 1}: {ctx.source} | 类型: {ctx.retrieval_type} | 分数: {ctx.score:.2f}]\n{ctx.content}"
            for index, ctx in enumerate(contexts)
        )

    @staticmethod
    def build_evidence(contexts: list[RetrievedContext]) -> list[DeepResearchEvidence]:
        evidence: list[DeepResearchEvidence] = []
        now = datetime.now(timezone.utc).isoformat()
        for index, ctx in enumerate(contexts, start=1):
            quote = " ".join(ctx.content.split())
            if len(quote) > 520:
                quote = f"{quote[:517]}..."
            evidence.append(
                DeepResearchEvidence(
                    evidence_id=f"E{index}",
                    source=ctx.source,
                    title=str(ctx.metadata.get("title", ctx.source)),
                    quote=quote,
                    confidence=min(max(coerce_float(ctx.score), 0.0), 1.0),
                    retrieved_at=str(ctx.metadata.get("retrieved_at", now)),
                    metadata={
                        "retrieval_type": ctx.retrieval_type,
                        **ctx.metadata,
                    },
                )
            )
        return evidence

    @staticmethod
    def _fallback_answer(
        evidence: list[DeepResearchEvidence],
        steps: list[DeepResearchStep],
    ) -> str:
        if evidence:
            lines = ["模型未返回可解析的结构化答案，以下是已检索到的证据摘要："]
            lines.extend(f"- {item.quote} [{item.evidence_id}]" for item in evidence[:5])
            return "\n".join(lines)
        return steps[-1].summary if steps else "未检索到足以回答问题的证据。"

    @staticmethod
    def validate_citations(
        answer: str,
        evidence: list[DeepResearchEvidence],
    ) -> tuple[str, list[str], list[str]]:
        valid_ids = {item.evidence_id for item in evidence}
        cited_ids: list[str] = []
        invalid_ids: list[str] = []

        def _replace(match: re.Match[str]) -> str:
            evidence_id = match.group(1).upper()
            if evidence_id in valid_ids:
                if evidence_id not in cited_ids:
                    cited_ids.append(evidence_id)
                return f"[{evidence_id}]"
            if evidence_id not in invalid_ids:
                invalid_ids.append(evidence_id)
            return ""

        cleaned = re.sub(r"\[(E\d+)\]", _replace, str(answer or ""), flags=re.IGNORECASE)
        cleaned = re.sub(r"[ \t]+([，。；：,.!?])", r"\1", cleaned).strip()
        warnings: list[str] = []
        if invalid_ids:
            warnings.append(f"已移除不存在的证据引用: {', '.join(invalid_ids)}")
        if evidence and not cited_ids:
            appendix = "；".join(f"[{item.evidence_id}] {item.title}" for item in evidence[:3])
            cleaned = f"{cleaned}\n\n证据索引：{appendix}".strip()
            cited_ids = [item.evidence_id for item in evidence[:3]]
            warnings.append("模型回答未包含有效证据引用，系统已附加证据索引。")
        if not evidence:
            warnings.append("本次研究未检索到可引用证据。")
        return cleaned or "未生成有效答案。", cited_ids, warnings

    @classmethod
    def calibrate_confidence(
        cls,
        model_confidence: float,
        evidence: list[DeepResearchEvidence],
        cited_ids: list[str],
    ) -> float:
        if not evidence:
            return 0.0
        model_score = min(max(coerce_float(model_confidence), 0.0), 1.0)
        average_evidence = sum(item.confidence for item in evidence) / len(evidence)
        sources = {item.source.split("#", 1)[0].rstrip("/").casefold() for item in evidence}
        diversity = min(len(sources) / max(settings.deepresearch_min_sources, 1), 1.0)
        citation_coverage = min(len(cited_ids) / max(min(len(evidence), 3), 1), 1.0)
        evidence_score = min(
            max(average_evidence * 0.55 + diversity * 0.25 + citation_coverage * 0.20, 0.0), 1.0
        )
        return min(model_score, evidence_score) if model_score else evidence_score * 0.8

    @staticmethod
    def format_evidence(evidence: list[DeepResearchEvidence]) -> str:
        return "\n".join(
            (
                f"[{item.evidence_id}] source={item.source} "
                f"title={item.title} confidence={item.confidence:.2f}\n"
                f"quote={item.quote}"
            )
            for item in evidence
        )

    # Backward-compatible aliases for existing callers/tests.
    @staticmethod
    def _clean_list(raw: Any) -> list[str]:
        return DeepResearchAgent.clean_list(raw)

    def _merge_contexts(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        return self.merge_contexts(contexts)

    @staticmethod
    def _format_contexts(contexts: list[RetrievedContext]) -> str:
        return DeepResearchAgent.format_contexts(contexts)


DeepSearchStep = DeepResearchStep
DeepSearchEvidence = DeepResearchEvidence
DeepSearchResult = DeepResearchResult
DeepSearchAgent = DeepResearchAgent
