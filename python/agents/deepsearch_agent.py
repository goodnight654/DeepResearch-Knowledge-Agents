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

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from services.hybrid_retriever import HybridRetriever, RetrievedContext


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
        self.llm = llm or ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_client_base_url,
            temperature=0,
        )
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
        plan = await self.plan_search(question)
        focus = str(plan.get("goal") or question)
        frontier = self.initial_frontier(plan, question)

        all_contexts: list[RetrievedContext] = []
        steps: list[DeepResearchStep] = []
        confidence = 0.0

        for iteration in range(1, self.max_iterations + 1):
            queries = frontier[: self.queries_per_iteration] or [question]
            round_contexts = await self.retrieve_round(queries)
            all_contexts = self.merge_contexts(all_contexts + round_contexts)

            summary = await self.summarize_round(question, focus, queries, all_contexts)
            gap_state = await self.analyze_gaps(question, focus, steps, summary, all_contexts)
            confidence = float(gap_state.get("confidence", confidence or 0.0))

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

            follow_up = self.clean_list(gap_state.get("follow_up_queries"))
            if self.should_finalize(iteration, gap_state, follow_up):
                break

            focus = str(gap_state.get("next_focus") or focus)
            frontier = follow_up

        contexts = self.merge_contexts(all_contexts)[: self.final_context_limit]
        evidence = self.build_evidence(contexts)
        executive_summary, answer, final_confidence = await self.compose_answer(
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
        )

    async def search(self, question: str) -> DeepResearchResult:
        return await self.research(question)

    async def plan_search(self, question: str) -> dict[str, Any]:
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
        for query in queries[: self.queries_per_iteration]:
            _, hits = await self.retriever.retrieve(
                query,
                top_k=self.contexts_per_query,
                per_query_limit=self.contexts_per_query,
                web_limit=min(self.contexts_per_query, settings.web_search_top_k),
            )
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
            f"当前可用上下文数: {len(contexts)}"
        )
        result = await self._load_json_response(
            GAP_ANALYSIS_PROMPT,
            payload,
            fallback={
                "answered": len(contexts) >= self.final_context_limit,
                "confidence": min(len(contexts) / max(self.final_context_limit, 1), 1.0),
                "next_focus": focus,
                "gaps": [],
                "follow_up_queries": [],
            },
        )
        result["gaps"] = self.clean_list(result.get("gaps"))
        result["follow_up_queries"] = self.clean_list(result.get("follow_up_queries"))
        return result

    async def compose_answer(
        self,
        question: str,
        plan: dict[str, Any],
        steps: list[DeepResearchStep],
        contexts: list[RetrievedContext],
        evidence: list[DeepResearchEvidence],
        fallback_confidence: float,
    ) -> tuple[str, str, float]:
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
        result = await self._load_json_response(
            FINAL_SYNTHESIS_PROMPT,
            payload,
            fallback={
                "executive_summary": steps[-1].summary if steps else "暂无搜索摘要。",
                "answer": steps[-1].summary if steps else "暂无答案。",
                "confidence": fallback_confidence,
            },
        )
        executive_summary = str(result.get("executive_summary") or "暂无搜索摘要。")
        answer = str(result.get("answer") or executive_summary)
        confidence = float(result.get("confidence", fallback_confidence or 0.0))
        return executive_summary, answer, min(max(confidence, 0.0), 1.0)

    @staticmethod
    def build_result(
        question: str,
        answer: str,
        executive_summary: str,
        contexts: list[RetrievedContext],
        steps: list[DeepResearchStep],
        evidence: list[DeepResearchEvidence],
        confidence: float,
    ) -> DeepResearchResult:
        return DeepResearchResult(
            question=question,
            answer=answer,
            executive_summary=executive_summary,
            contexts=contexts,
            steps=steps,
            evidence=evidence,
            confidence=confidence,
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

        cleaned = str(resp.content).strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            return fallback
        return data if isinstance(data, dict) else fallback

    @staticmethod
    def clean_list(raw: Any) -> list[str]:
        if not isinstance(raw, list):
            return []
        values = [str(item).strip() for item in raw if item is not None and str(item).strip()]
        return list(dict.fromkeys(values))

    def merge_contexts(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        deduped: dict[tuple[str, str], RetrievedContext] = {}
        for ctx in contexts:
            key = (ctx.source, ctx.content[:160])
            existing = deduped.get(key)
            if existing is None or ctx.score > existing.score:
                deduped[key] = ctx
        merged = list(deduped.values())
        merged.sort(key=lambda item: item.score, reverse=True)
        return merged

    @staticmethod
    def format_contexts(contexts: list[RetrievedContext]) -> str:
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
                    confidence=min(max(float(ctx.score), 0.0), 1.0),
                    retrieved_at=str(ctx.metadata.get("retrieved_at", now)),
                    metadata={
                        "retrieval_type": ctx.retrieval_type,
                        **ctx.metadata,
                    },
                )
            )
        return evidence

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
