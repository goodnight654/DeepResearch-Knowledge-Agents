"""
问答 Agent — 混合检索 + 答案生成

重构后职责:
  1. 意图识别
  2. 调用统一 HybridRetriever 检索上下文
  3. 基于上下文生成最终答案

说明:
  QAAgent 聚焦“单轮问答”，更复杂的多轮研究编排由 DeepResearchAgent 负责。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from services.hybrid_retriever import HybridRetriever, RetrievedContext


class QueryIntent(str, Enum):
    FACTOID = "factoid"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"


@dataclass
class QAResult:
    question: str
    answer: str
    contexts: list[RetrievedContext]
    intent: QueryIntent
    confidence: float
    reasoning_steps: list[str] = field(default_factory=list)


INTENT_PROMPT = """\
你是一个查询意图分类器。根据用户问题，返回意图类别（只返回类别名）：
- factoid: 事实型（谁/什么/哪里/何时）
- analytical: 分析型（为什么/怎么理解）
- comparative: 对比型（A和B有什么区别）
- procedural: 流程型（怎么做/步骤）
- exploratory: 探索型（有哪些/概述）
"""

ANSWER_PROMPT = """\
你是一个专业的企业知识问答助手。根据检索到的上下文信息回答用户问题。

要求：
1. 答案必须基于提供的上下文，不要编造
2. 如果上下文信息不足，明确告知用户
3. 引用信息来源（如 [来源: xxx]）
4. 如果涉及多个信息源，综合分析后给出结论
5. 保持专业、准确、简洁
"""


class QAAgent:
    """
    单轮问答 Agent

    工作流:
      query → intent_classify → hybrid_retrieve → generate_answer
    """

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

    async def answer(self, question: str) -> QAResult:
        intent = await self._classify_intent(question)
        analysis, contexts = await self.retriever.retrieve(
            question,
            top_k=settings.qa_context_limit,
            per_query_limit=max(2, settings.qa_context_limit // 2),
        )
        answer_text, reasoning = await self._generate_answer(question, contexts, intent, analysis.queries)

        return QAResult(
            question=question,
            answer=answer_text,
            contexts=contexts,
            intent=intent,
            confidence=self._calc_confidence(contexts),
            reasoning_steps=reasoning,
        )

    async def _classify_intent(self, question: str) -> QueryIntent:
        messages = [
            SystemMessage(content=INTENT_PROMPT),
            HumanMessage(content=question),
        ]
        try:
            resp = await self.llm.ainvoke(messages)
            raw = str(resp.content).strip().lower()
        except Exception:
            return QueryIntent.FACTOID

        for intent in QueryIntent:
            if intent.value in raw:
                return intent
        return QueryIntent.FACTOID

    async def _generate_answer(
        self,
        question: str,
        contexts: list[RetrievedContext],
        intent: QueryIntent,
        queries: list[str],
    ) -> tuple[str, list[str]]:
        context_text = "\n\n".join(
            f"[来源 {i + 1}: {ctx.source} | 类型: {ctx.retrieval_type} | 分数: {ctx.score:.2f}]\n{ctx.content}"
            for i, ctx in enumerate(contexts)
        )
        reasoning_steps = [
            f"识别问题意图: {intent.value}",
            f"生成检索查询: {queries[:3]}",
            f"检索到 {len(contexts)} 条相关上下文",
            f"向量检索: {sum(1 for ctx in contexts if ctx.retrieval_type == 'vector')} 条",
            f"图谱检索: {sum(1 for ctx in contexts if ctx.retrieval_type == 'graph')} 条",
        ]

        messages = [
            SystemMessage(content=ANSWER_PROMPT),
            HumanMessage(content=f"上下文信息:\n{context_text}\n\n用户问题: {question}"),
        ]
        try:
            resp = await self.llm.ainvoke(messages)
            answer = str(resp.content)
        except Exception:
            answer = "暂时无法生成答案，请检查模型或检索服务配置。"
        reasoning_steps.append("答案生成完成")
        return answer, reasoning_steps

    @staticmethod
    def _calc_confidence(contexts: list[RetrievedContext]) -> float:
        if not contexts:
            return 0.0
        avg_score = sum(ctx.score for ctx in contexts) / len(contexts)
        return min(avg_score, 1.0)
