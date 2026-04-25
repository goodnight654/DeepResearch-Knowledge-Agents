# 简历成稿（中文易懂版）

> 这是一版可直接粘贴到简历的项目描述，重点先讲“做了什么、解决什么问题”，英文技术名只作为补充，避免项目经历看起来像术语堆砌。

---

## 版本A：中文简历（推荐）

### 项目名称
**智能资料研究助手（本地知识库 + 联网深度研究）**

### 项目时间
2026.01 - 2026.04

### 项目角色
核心开发（架构设计 + Python主实现）

### 技术栈
Python 后端、智能体流程编排、向量检索、知识图谱、网页搜索、FastAPI、Docker

### 项目简介（2-3行）
设计并实现一个“能读公司资料、能查本地知识库、也能联网补充证据”的智能研究助手。普通问题可以快速从本地文档中回答；复杂开放问题会自动拆解、分多轮检索、读取网页正文、补齐信息缺口，并给出带证据编号的结论。系统还支持运行轨迹回放，便于解释每一步为什么这样检索和回答。

### 核心贡献（简历条目）
- 搭建“资料入库、知识提取、本地问答、增量更新、深度研究”完整链路，让系统既能处理企业文档，也能持续维护知识库。
- 设计复杂问题的多轮研究流程：先拆解问题，再反复检索资料、总结阶段结论、判断信息缺口，最后综合生成答案，避免单次问答证据不足。
- 实现“本地知识库 + 联网搜索”的统一检索能力：优先召回本地文档和知识图谱，需要外部资料时再搜索网页，并进一步读取网页正文、清洗分块后作为证据。
- 设计证据级引用机制，为每条关键依据生成 `[E1]` 这类证据编号，让最终回答能追溯到具体文档或网页来源。
- 实现运行轨迹保存和可视化回放，把每次问题的拆解、检索、补查、总结和最终回答保存下来，方便调试、复盘和面试演示。
- 新增深度研究接口和运行记录接口，形成“快速问答 + 深度研究”的双通道服务：简单问题快速回答，复杂问题走多轮研究。

### 技术关键词（面试追问时展开）
智能体编排（LangGraph）/ 本地知识库问答（RAG）/ 知识图谱（Neo4j）/ 向量检索（ChromaDB 或 PGVector）/ 联网搜索（DuckDuckGo、Tavily、SerpAPI）/ 证据引用 / 运行轨迹回放

### 结果与价值（可选）
- 架构层面：从“单次问答工具”升级为“可解释的研究型智能体系统”。
- 工程层面：支持证据追溯、运行回放和策略迭代，方便后续评测与优化。
- 面试展示层面：不仅能展示最终答案，还能展示系统每一轮为什么继续查、查了什么、为什么停止。

---

## 版本B：英文简历（投外企可用）

### Project
**DeepResearch Knowledge Hub (Local Knowledge Base + Web DeepResearch Agent)**

### Role
Core Engineer (Architecture + Python Implementation)

### Stack
Python, LangGraph, LangChain, FastAPI, Neo4j, ChromaDB/PGVector, Kafka, Docker

### Highlights
- Built a multi-agent knowledge system covering document parsing, knowledge extraction, QA, and incremental updates with LangGraph orchestration.
- Refactored DeepResearch into a multi-stage loop (`plan -> retrieve -> summarize -> gap -> synthesize`) with explicit routing and stopping conditions.
- Implemented a unified retrieval layer combining vector, graph, web page reading, and pluggable web search providers (duckduckgo/tavily/serpapi/disabled).
- Designed run-level trace persistence (`input/trace/output/status`) and shipped a trace visualization UI for replay and debugging.
- Added DeepResearch and trace APIs (`/api/qa/deep-research`, `/api/qa/runs`, `/api/qa/runs/{run_id}`), enabling explainable and observable agent execution.

---

## 面试时口播版本（30秒）

我做的是一个智能资料研究助手。它既能读取企业内部文档、建立本地知识库，也能在问题比较复杂时联网搜索并读取网页正文。和普通问答不同，我把复杂问题拆成多轮研究流程：先拆题，再查本地资料和网页，发现证据不够就继续补查，最后输出带证据编号的结论。同时我保存了每次运行轨迹，可以在页面上回放系统每轮查了什么、为什么继续、为什么停止，所以它不是黑盒问答，而是一个可解释、可调试的研究型智能体。

## 更短版本（15秒）

我做的是一个“本地知识库 + 联网搜索”的研究型智能体。简单问题直接查内部资料回答，复杂问题会自动拆解、多轮检索、读取网页正文，并把最终结论绑定到证据来源，同时支持运行轨迹回放，方便解释和调试。
