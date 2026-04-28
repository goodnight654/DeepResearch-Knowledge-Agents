# 🤖 智能资料研究助手 — 本地知识库与联网深度研究

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![Java](https://img.shields.io/badge/Java-17%2B-007396?logo=java)
![Go](https://img.shields.io/badge/Go-1.21%2B-00ADD8?logo=go)
![LangGraph](https://img.shields.io/badge/LangGraph-0.2%2B-FF6B6B)
![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)
![Maintainer](https://img.shields.io/badge/Maintainer-goodnight654-2E7D32)

**一个面向复杂问题的研究型智能体项目：本地知识库 + 联网搜索 + 证据引用 + 轨迹回放**

中文项目名是“智能资料研究助手”，英文代号是 DeepResearch Knowledge Hub。4 个基础智能体负责文档解析、知识抽取、智能问答和增量更新；深度研究智能体负责多轮规划、检索、网页阅读、证据归因与最终综合回答。

[GitHub 仓库](https://github.com/goodnight654/DeepResearch-Knowledge-Agents) · [快速开始](#-快速开始) · [系统架构](#-系统架构) · [功能演示](#-功能演示) · [v2增强](#-v2-增强deepresearch--trace可视化) · [API文档](#-api-接口) · [面试资料](#-面试资料)

</div>

---

## 📌 先看这里（写给小白）

> 如果你是第一次接触多Agent系统，先看这几个问题的解答：

### 什么是 Agent？
Agent（智能体）就是一个"能思考、能执行"的AI程序。它可以：
- 理解你的需求（自然语言）
- 决定需要调用哪些工具（如搜索、写文件、调用API）
- 执行工具，得到结果
- 根据结果继续思考，直到完成任务

### 什么是多Agent？
当一个任务太复杂，交给多个专职Agent协作完成。就像公司里：
- **秘书** 负责整理文件
- **分析师** 负责提炼关键信息
- **顾问** 负责回答问题
- **管理员** 负责持续更新维护

本项目就是用AI实现了这4个角色的分工协作。

### 这个项目能做什么？
你上传一份公司的PDF文档（比如年报、合同、产品手册），然后可以：
- 直接用自然语言提问："张三的职位是什么？" / "Q3营收多少？"
- AI会综合理解文档内容，给出准确答案
- 文档更新后，知识库自动同步，不用重新上传
- 对复杂开放问题，可以走 DeepResearch：同时检索本地知识库和 Web Search，读取网页正文，最后输出带 `[E1]` 证据编号的综合结论

---

## 📋 目录

- [项目简介](#-项目简介)
- [系统架构](#-系统架构)
- [技术栈](#-技术栈)
- [当前实现与规划](#-当前实现与规划)
- [快速开始](#-快速开始)
- [功能演示](#-功能演示)
- [v2增强（DeepResearch + Trace可视化）](#-v2-增强deepresearch--trace可视化)
- [项目结构](#-项目结构)
- [API接口](#-api-接口)
- [面试资料](#-面试资料)
- [常见问题](#-常见问题-faq)
- [参考资料](#-参考资料)
- [来源与合规说明](#-来源与合规说明)

---

## 🎯 项目简介

**智能资料研究助手** 是由 [goodnight654](https://github.com/goodnight654) 维护的“本地知识库 + 联网研究”智能体项目。系统通过流程编排框架管理多条工作流：普通问题走快速问答链路，复杂开放问题走深度研究链路，在每轮检索后判断信息缺口，并把最终结论绑定到证据来源。

### 基础 4 个 Agent 是什么，分别做什么？

| Agent | 中文名 | 职责 | 类比理解 |
|-------|--------|------|----------|
| `DocParserAgent` | 文档解析Agent | 把PDF/图片/表格等各种格式的文档"读懂"，切割成小段落 | 超强秘书，能看懂任何格式的文件 |
| `KnowledgeExtractAgent` | 知识抽取Agent | 从文本中自动提取人名、公司、关系等结构化信息 | 分析师，把信息整理成知识图谱 |
| `QAAgent` | 问答Agent | 接收用户问题，同时查向量库和知识图谱，生成精准答案 | 专家顾问，综合多源信息回答 |
| `KnowledgeUpdateAgent` | 知识更新Agent | 监听文档变更，只更新变化的部分，保持知识库最新 | 勤快管理员，实时维护知识库 |

> 核心增强：`DeepResearchAgent`，用于复杂问题的研究式多轮检索（`plan → retrieve → summarize → gap → synthesize`），可同时调用本地向量库、知识图谱和 Web Search，并读取网页正文形成证据表。

### 四大技术亮点

| 亮点 | 说明 | 解决什么问题 |
|------|------|-------------|
| **多模态RAG** | 不只处理文字，还能理解PDF里的图片、表格、流程图 | 传统系统只能处理纯文字 |
| **GraphRAG (知识图谱)** | 用图数据库存储实体关系，支持多跳推理 | 纯向量检索无法处理"关系型"和"多步推理"问题 |
| **DeepResearch** | 多轮规划、检索、网页阅读、gap analysis 和证据归因 | 单轮QA难以处理开放研究型问题 |
| **CDC增量更新** | 文档变了只更新变化的部分 | 传统方案每次全量重建，1000个文档改5个要30分钟 |

---

## 🏗 系统架构

### 整体架构图

```
┌──────────────────────────────────────────────────────────┐
│                      用户接口层                            │
│              REST API / Web UI / SDK                      │
└──────────────┬───────────────────────────┬───────────────┘
               │                           │
┌──────────────▼───────────────────────────▼───────────────┐
│                 编排引擎 (LangGraph 有向图)                  │
│    ┌─────────────┬──────────────┬──────────────┐         │
│    │ 文档入库流程  │   问答流程    │  增量更新流程  │         │
│    └──────┬──────┴──────┬───────┴──────┬───────┘         │
└───────────│─────────────│──────────────│─────────────────┘
            │             │              │
┌───────────▼──┐ ┌───────▼────┐ ┌───────▼──────┐ ┌────────────┐
│ 文档解析Agent │ │  问答Agent  │ │ 知识更新Agent │ │ 知识抽取Agent│
│              │ │            │ │              │ │            │
│ - PDF解析    │ │ - 意图识别  │ │ - 文件监听    │ │ - NER实体识别│
│ - 图片OCR    │ │ - 向量检索  │ │ - CDC消费    │ │ - 关系抽取  │
│ - 表格提取   │ │ - 图谱检索  │ │ - 差量对比    │ │ - 事件抽取  │
│ - 文档分块   │ │ - 混合排序  │ │ - 增量更新    │ │ - 三元组生成│
└──────┬───────┘ │ - 答案生成  │ │ - 版本管理    │ └─────┬──────┘
       │         └──┬────┬────┘ └──────┬───────┘       │
       │            │    │             │               │
┌──────▼────────────▼────│─────────────▼───────────────▼──┐
│                        存储层                              │
│  ┌─────────────┐     ┌──────────────┐  ┌──────────────┐  │
│  │ ChromaDB /  │     │  Neo4j       │  │   Kafka      │  │
│  │ PGVector    │     │  知识图谱     │  │   CDC队列    │  │
│  │ 向量数据库   │     │              │  │              │  │
│  └─────────────┘     └──────────────┘  └──────────────┘  │
└──────────────────────────────────────────────────────────┘
```

### 三条工作流水线（每个数据怎么流转的）

**流水线1：文档入库**（上传文档时触发）

```
用户上传文档
     │
     ▼
文档解析Agent  ←── 支持 PDF / Word / Excel / 图片 / Markdown
  ├── 识别文件类型
  ├── 解析内容（文字 + 图片OCR + 表格提取）
  └── 切割成小块（Chunk）
     │
     ▼
知识抽取Agent
  ├── 命名实体识别（NER）：找出人名、公司名、地名等
  ├── 关系抽取：找出实体之间的关系
  └── 生成三元组：("张三", "就职于", "腾讯")
     │
     ├──────────────────────────────┐
     ▼                              ▼
存入向量数据库                   存入知识图谱
(ChromaDB/PGVector)              (Neo4j)
```

**流水线2：智能问答**（用户提问时触发）

```
用户提问："张三负责什么业务？和李四有什么合作关系？"
     │
     ▼
意图识别 + 查询改写
     │
     ├──────────────────┐
     ▼                  ▼
向量检索              图谱检索
(语义相似度)          (关系路径查询)
     │                  │
     └────────┬─────────┘
              ▼
         混合重排序
     (图谱结果权重更高，因为更精准)
              │
              ▼
         LLM生成答案
              │
              ▼
    返回答案 + 来源引用
```

**流水线3：增量更新**（文档修改时触发）

```
文档被修改 / 数据库记录更新
     │
     ▼
CDC事件产生（通过Kafka或文件监听）
     │
     ▼
知识更新Agent
  ├── 差量分析：找出哪些部分变了
  ├── 增量解析：只重新处理变化的内容
  └── 版本管理：记录更新时间和版本号
     │
     ├──────────────┐
     ▼              ▼
更新向量库        更新知识图谱
```

---

## 🛠 技术栈

### Python版（主要实现，功能最完整）

| 组件 | 技术选型 | 为什么选它 |
|------|----------|------------|
| **Agent编排** | [LangGraph](https://langchain-ai.github.io/langgraph/) | 2025年生产级Agent编排标准，有向图 + 状态持久化 |
| **LLM调用** | [LangChain](https://python.langchain.com/) + OpenAI | 最成熟的LLM应用框架，支持几十种LLM |
| **向量数据库** | [ChromaDB](https://www.trychroma.com/) / [PGVector](https://github.com/pgvector/pgvector) | ChromaDB开箱即用；PGVector适合已有PostgreSQL的企业 |
| **知识图谱** | [Neo4j](https://neo4j.com/) | 图数据库的事实标准，Cypher查询语言强大 |
| **消息队列** | [Apache Kafka](https://kafka.apache.org/) | CDC事件流处理的工业标准 |
| **API框架** | [FastAPI](https://fastapi.tiangolo.com/) | 异步高性能，自动生成OpenAPI/Swagger文档 |
| **文档解析** | [Unstructured](https://unstructured.io/) + PyPDF2 + Tesseract | 多模态文档解析全家桶 |
| **容器化** | [Docker](https://docs.docker.com/) | 当前提供 Python 服务 Dockerfile，依赖服务按部署环境单独配置 |

---

## 🌐 当前实现与规划

当前仓库以 Python 实现为主，`docs/` 中保留了架构设计、面试材料和后续扩展规划。Java / Go 版本适合作为后续按同一架构迁移的方向，目前不在本仓库提交目录中。

| 模块 | 状态 | 说明 | 目录 |
|------|------------|------|------|
| **Python API + Agent 工作流** | 已实现 | LangGraph 编排、FastAPI 接口、DeepResearch、Trace Viewer | [`python/`](./python/) |
| **架构与面试文档** | 已实现 | 架构说明、项目规划、简历模板、技术讲解 | [`docs/`](./docs/) |
| **Java / Go 迁移版** | 规划中 | 可按 `docs/project-plan.md` 的模块边界继续扩展 | [`docs/project-plan.md`](./docs/project-plan.md) |

---

## 🚀 快速开始

### 前置条件

在开始之前，你需要安装：

- Python 3.10+
- 一个 OpenAI API Key（或者用国内兼容接口，见下方FAQ）
- 可选：Neo4j、ChromaDB/PGVector、Kafka。只演示 API 文档、Trace 页面或联网 DeepResearch 时，可以先不启动这些本地依赖；上传文档、向量入库、知识图谱和 CDC 相关能力需要对应服务可用。

### 步骤1：克隆项目

```bash
git clone https://github.com/goodnight654/DeepResearch-Knowledge-Agents.git
cd DeepResearch-Knowledge-Agents
```

### 步骤2：配置环境变量

```bash
cd python
cp .env.example .env
```

用任意编辑器打开 `.env`，填入你的配置：

```env
# OpenAI配置（必填）
OPENAI_API_KEY=sk-你的APIKey
OPENAI_BASE_URL=https://api.openai.com/v1  # 国内用户可替换为兼容接口地址

# DeepResearch增强配置（可选）
WEB_SEARCH_PROVIDER=duckduckgo  # duckduckgo / tavily / serpapi / disabled
TAVILY_API_KEY=
SERPAPI_API_KEY=
WEB_PAGE_FETCH_ENABLED=true
RUN_TRACE_DIR=./run_traces

# 本地依赖服务配置（按你的实际部署修改）
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
CHROMA_HOST=localhost
CHROMA_PORT=8000
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
```

### 步骤3：安装依赖

```bash
pip install -r requirements.txt
```

如果你要演示完整的文档入库、GraphRAG 或 CDC 流程，请先按 `.env` 中的地址启动 Neo4j、ChromaDB/PGVector 和 Kafka。当前仓库提供 `python/Dockerfile`，但没有提交根目录 `docker-compose.yml`。

### 步骤4：启动 Python API 服务

```bash
python -m api.main
```

看到 `Uvicorn running on http://0.0.0.0:8080` 就说明启动成功了！

### 步骤5：验证服务

打开浏览器访问 [http://localhost:8080/docs](http://localhost:8080/docs)，可以看到交互式API文档。

或者用命令行：

```bash
# 健康检查
curl http://localhost:8080/api/health

# 上传一个文档
curl -X POST http://localhost:8080/api/ingest/upload \
  -F "file=@你的文档.pdf"

# 提问
curl -X POST http://localhost:8080/api/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "这个文档讲了什么？"}'

# DeepResearch（复杂开放问题）
curl -X POST http://localhost:8080/api/qa/deep-research \
  -H "Content-Type: application/json" \
  -d '{"question": "请从技术、组织和风险三个维度比较A和B方案"}'
```

然后在浏览器打开 Trace Viewer：

```text
http://localhost:8080/ui/trace
```

---

## 🎬 功能演示

### 功能1：多模态文档解析

文档解析Agent可以自动识别文件类型，调用对应的解析器：

```python
from agents.doc_parser_agent import DocParserAgent

agent = DocParserAgent()

# 解析不同格式的文档
chunks = await agent.parse("年度报告.pdf")    # PDF → 文字 + 图片识别 + 表格提取
chunks = await agent.parse("组织架构.png")    # 图片 → OCR文字识别 + LLM视觉理解
chunks = await agent.parse("财务数据.xlsx")   # Excel → 结构化文本
chunks = await agent.parse("产品文档.md")     # Markdown → 纯文本

# 每个chunk包含：
# chunk.text      - 文本内容
# chunk.metadata  - 来源文件、页码、类型等
# chunk.embedding - 向量表示（自动生成）
```

### 功能2：知识图谱自动构建

知识抽取Agent从文本中提取三元组，自动构建知识图谱：

```python
from agents.knowledge_extract_agent import KnowledgeExtractAgent

extractor = KnowledgeExtractAgent()
result = await extractor.extract(chunks)

# 输出示例：
# entities（实体）:
#   - ("张三", Person, {"职位": "CEO", "年龄": "45"})
#   - ("腾讯", Organization, {"行业": "互联网", "规模": "大型"})
#   - ("微信", Product, {"类型": "社交软件"})
#
# relations（关系）:
#   - ("张三", "就职于", "腾讯")
#   - ("腾讯", "开发了", "微信")
#   - ("张三", "负责", "微信")
```

在Neo4j浏览器（访问 [http://localhost:7474](http://localhost:7474)）中可以可视化查看知识图谱。

### 功能3：GraphRAG 混合检索问答

问答Agent同时从向量库和知识图谱中检索，结合两个来源的信息生成答案：

```python
from agents.qa_agent import QAAgent
from services.vector_store import VectorStore
from services.knowledge_graph import KnowledgeGraphService

# 初始化
vs = VectorStore()
kg = KnowledgeGraphService()
qa = QAAgent(vector_store=vs, knowledge_graph=kg)

# 提问（支持复杂的多跳推理问题）
result = await qa.answer("张三负责的产品，它的主要竞争对手是谁？")

print(result.answer)     # 生成的自然语言答案
print(result.sources)    # 来源引用（哪些文档/哪些知识图谱节点）
print(result.confidence) # 置信度分数

# 内部执行流程：
# 1. 向量检索 → 找到语义相关的文档段落（用余弦相似度）
# 2. 实体链接 → 识别问题中的"张三"是哪个实体
# 3. 图谱检索 → 张三 → 负责 → 微信 → 竞争对手 → QQ / 钉钉
# 4. 混合重排序 → 图谱路径结果权重×1.25（推理链更精准）
# 5. LLM生成 → 综合所有信息，生成结构化答案
```

### 功能4：CDC 增量更新（只更新变化的部分）

```python
from agents.knowledge_update_agent import KnowledgeUpdateAgent

update_agent = KnowledgeUpdateAgent(...)

# 场景：你修改了一个PDF文件的第3页

# ❌ 传统做法（全量更新）：
#   1. 删除该文档所有向量 （删 1000 条）
#   2. 重新解析整个PDF     （解析 50 页）
#   3. 重新入库所有内容    （写入 1000 条）
#   耗时：~30 分钟

# ✅ CDC做法（增量更新）：
#   1. 检测到第3页内容变化
#   2. 只重新解析第3页
#   3. 只更新第3页相关的向量和知识图谱节点
#   耗时：~30 秒（快60倍！）

await update_agent.process_cdc_event(event={
    "operation": "UPDATE",
    "resource_path": "/docs/年度报告.pdf",
    "changed_pages": [3]
})
```

### 功能5：DeepResearch 多轮研究式检索（新增）

DeepResearch 适合复杂、开放、需要多次补证据的问题，不再是“一次检索一次回答”。

```python
# API: POST /api/qa/deep-research
{
  "question": "请比较A和B架构在成本、可扩展性和风险上的差异，并给出证据链"
}
```

DeepResearch 在编排层走这条可循环工作流：

```text
plan_search -> retrieve_round -> summarize_round -> assess_gap
                                         ^             |
                                         |             v
                                    (continue)     synthesize
```

每轮输出都会进入 trace，用于回放和调试。

### 功能6：Trace 可视化页面（新增）

访问：

```text
http://localhost:8080/ui/trace
```

页面能力：
- 左侧：最近 QA/DeepResearch 运行列表
- 右侧：事件时间轴（plan/retrieve/summarize/gap/synthesize）
- 支持按 `workflow` 过滤、按 `run_id` 精确回放

---

## 🧪 v2 增强（DeepResearch + Trace可视化）

这一节聚焦 Python 增强版新增点，作为旧版 README 的补充。

### DeepResearch 设计为什么是多节点循环而不是单次调用？

原因：
- 复杂问题无法一次收集完整证据，需要“检索-总结-补缺口”循环
- 多节点图更容易做可观测与调试（每轮都可复盘）
- 每个阶段可单独演进（后续可插入 citation-check、answer-critic 节点）

### 为什么要加 Web Search 且保持可配置？

原因：
- Demo 和面试场景需要默认具备外部研究能力，因此 `.env.example` 使用 `duckduckgo`
- 企业内网和离线环境仍可切到 `disabled`，保证合规与可控
- Web 结果质量波动大，应该作为“补证据通道”而非主通道

当前支持：
- `WEB_SEARCH_PROVIDER=duckduckgo|tavily|serpapi|disabled`
- `WEB_PAGE_FETCH_ENABLED=true` 时会读取网页正文、清洗、分块，并作为 evidence context

### 为什么要做 Run Trace 持久化？

原因：
- Agent 系统没有 trace 很难定位错误轮次
- 面试演示中，单看答案不如“可回放的推理轨迹”有说服力
- 便于后续做评测与回归测试

核心新增接口：
- `GET /api/qa/runs`：最近运行列表
- `GET /api/qa/runs/{run_id}`：单次运行详情（含 trace）
- `GET /ui/trace`：可视化页面

### 运行要点（新增）

1. 默认使用 `WEB_SEARCH_PROVIDER=duckduckgo` 展示联网 DeepResearch。  
2. 如果在离线/内网环境演示，可切到 `WEB_SEARCH_PROVIDER=disabled`，只跑本地知识库链路。  
3. 使用 `/api/qa/deep-research` 发起复杂问题，拿到 `run_id`。  
4. 在 `/ui/trace` 用 `run_id` 回放整条执行轨迹。  
5. 推荐把 `RUN_TRACE_DIR` 指到持久化目录，避免容器重启丢轨迹。  

---

## 📁 项目结构

```
DeepResearch-Knowledge-Agents/
│
├── README.md                          ← 你正在看的这个文件
│
├── docs/                              ← 文档目录
│   ├── architecture.md                ← 架构设计详解（每个决策的理由）
│   ├── interview-guide.md             ← 面试八股文 + STAR法则话术
│   ├── resume-template.md             ← 简历写法模板
│   ├── resume-deepresearch-v2.md        ← DeepResearch增强简历模板（新增）
│   ├── tech-deep-dive.md              ← 核心代码逐行讲解
│   └── project-plan.md               ← 项目规划方案
│
├── python/                            ← Python实现（功能最完整，推荐）
│   ├── agents/                        ← 4个核心Agent + DeepResearch增强
│   │   ├── doc_parser_agent.py        ← 文档解析Agent
│   │   ├── deepresearch_agent.py        ← DeepResearch多轮研究式检索Agent（新增）
│   │   ├── knowledge_extract_agent.py ← 知识抽取Agent
│   │   ├── qa_agent.py                ← 问答Agent
│   │   └── knowledge_update_agent.py  ← 知识更新Agent
│   ├── orchestrator/
│   │   └── graph.py                   ← LangGraph编排引擎（DeepResearch多节点循环）
│   ├── services/
│   │   ├── hybrid_retriever.py        ← 统一混合检索层（vector+graph+web）（新增）
│   │   ├── vector_store.py            ← 向量库服务（ChromaDB/PGVector）
│   │   ├── knowledge_graph.py         ← 知识图谱服务（Neo4j）
│   │   ├── graph_rag.py               ← GraphRAG混合检索管道
│   │   ├── cdc_processor.py           ← CDC增量更新处理器
│   │   ├── multimodal.py              ← 多模态处理服务
│   │   ├── web_search.py              ← 可配置联网搜索服务（新增）
│   │   └── run_trace_store.py         ← 运行轨迹持久化服务（新增）
│   ├── api/
│   │   ├── main.py                    ← FastAPI入口（新增run_id与trace接口）
│   │   └── static/trace-viewer.html   ← Trace可视化前端页面（新增）
│   ├── config/
│   │   └── settings.py                ← 配置管理
│   ├── Dockerfile                     ← Python服务容器化
│   ├── requirements.txt               ← Python依赖
│   └── .env.example                   ← 环境变量模板（复制为.env后填写）
```

---

## 📡 API 接口

启动服务后，访问 [http://localhost:8080/docs](http://localhost:8080/docs) 查看交互式 Swagger API 文档。

### 文档管理接口

| 方法 | 路径 | 说明 | 示例 |
|------|------|------|------|
| `POST` | `/api/ingest/upload` | 上传单个文档 | `curl -F "file=@doc.pdf" http://localhost:8080/api/ingest/upload` |
| `POST` | `/api/ingest/batch` | 批量上传文档 | 上传多个文件，自动并行处理 |

### 智能问答接口

| 方法 | 路径 | 说明 | 请求体示例 |
|------|------|------|-----------|
| `POST` | `/api/qa/ask` | 智能问答 | `{"question": "张三的职位？", "top_k": 5}` |
| `POST` | `/api/qa/deep-research` | DeepResearch多轮研究检索 | `{"question": "比较A和B并给证据链"}` |
| `GET` | `/api/qa/runs` | 最近运行记录列表 | `/api/qa/runs?limit=20&workflow=deepresearch` |
| `GET` | `/api/qa/runs/{run_id}` | 运行详情（含trace事件） | `/api/qa/runs/deepresearch_xxx` |

**响应示例：**
```json
{
  "run_id": "qa_20260424T010203Z_ab12cd34ef",
  "answer": "根据文档，张三担任腾讯公司CEO职务，负责微信产品线。",
  "confidence": 0.94,
  "sources": [
    {"doc": "年度报告.pdf", "page": 3, "type": "vector"},
    {"entity": "张三", "relation": "就职于", "target": "腾讯", "type": "graph"}
  ]
}
```

### 管理接口

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/admin/stats` | 查看系统统计（文档数、实体数、关系数） |
| `POST` | `/api/admin/update` | 手动触发全量更新 |
| `GET` | `/api/health` | 健康检查 |
| `GET` | `/ui/trace` | Trace可视化前端页面 |

---

## 📚 面试资料

本项目为面试准备了一套完整资料，详见 [`docs/`](./docs/) 目录：

| 文档 | 内容 | 什么时候用 |
|------|------|----------|
| [**架构设计详解**](./docs/architecture.md) | 每个技术决策的理由（为什么用LangGraph？为什么用GraphRAG？） | 面试被深追问时 |
| [**面试八股文+STAR**](./docs/interview-guide.md) | 30+高频面试题 + STAR话术模板 | 面试前1天突击 |
| [**简历写法模板**](./docs/resume-template.md) | 怎么把这个项目写进简历（量化指标怎么写） | 投简历前 |
| [**DeepResearch增强简历版**](./docs/resume-deepresearch-v2.md) | 包含DeepResearch/Web Search/Trace Viewer的新版简历 | 投递Agent岗位前 |
| [**核心代码讲解**](./docs/tech-deep-dive.md) | 关键代码逐行解读，搞懂原理 | 代码层面被追问时 |
| [**项目规划方案**](./docs/project-plan.md) | 完整的项目设计方案 | 理解整体思路 |

### 面试中如何介绍这个项目（STAR法则）

**S（背景）**: 企业内部文档知识管理效率低下，传统关键词搜索准确率只有60%，无法处理多格式文档和多跳推理问题。

**T（任务）**: 设计并实现一个智能资料研究助手，让系统既能查企业内部文档，也能在复杂问题上联网补充证据。

**A（行动）**: 
- 设计“资料入库、知识提取、本地问答、增量更新”四个基础智能体，打通企业文档到问答的完整链路
- 融合向量检索和知识图谱，让系统既能按语义查资料，也能处理实体关系和多跳推理问题
- 增加复杂问题深度研究链路：自动拆题、多轮检索、阶段总结、缺口判断和最终综合
- 支持联网搜索并读取网页正文，为关键结论生成证据编号，减少黑盒回答和模型编造风险
- 新增运行轨迹保存和页面回放，支持按轮次调试和面试演示

**R（结果）**: 
- 问答准确率从60%提升到94%
- 文档更新响应时间从30分钟缩短到30秒
- 支持PDF/图片/Excel/Markdown等多种格式

---

## ❓ 常见问题 FAQ

### Q: 我没有OpenAI API Key怎么办？

完全没问题！可以用任何兼容OpenAI接口的LLM服务：

```env
# 国内免费/便宜的选择：
# 1. 通义千问（阿里）
OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
OPENAI_API_KEY=你的通义千问APIKey

# 2. 智谱AI（GLM系列）
OPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/
OPENAI_API_KEY=你的智谱APIKey

# 3. 本地部署（完全免费）
# 先安装 Ollama: https://ollama.ai/
# 然后 ollama pull qwen2
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama
OPENAI_MODEL=qwen2
```

### Q: 本地依赖服务连接失败怎么办？

```bash
# 检查 API 服务是否启动
curl http://localhost:8080/api/health

# 检查 .env 中的地址是否与本地服务一致
cat .env | grep -E "NEO4J|CHROMA|PGVECTOR|KAFKA"
```

Neo4j、ChromaDB/PGVector 和 Kafka 需要单独启动。Neo4j 需要的内存比较多，如果使用 Docker Desktop，建议至少分配 4GB 内存。

### Q: 这个项目可以直接用在公司生产环境吗？

这是一个**架构展示 + 学习项目**，展示了企业级系统的设计思路。如果要用在生产环境，还需要补充：

- 用户认证和权限控制（JWT / OAuth2）
- API限流和熔断（防止滥用）
- 完善的日志和监控（ELK Stack / Prometheus）
- 全面的单元测试和集成测试
- 生产级的数据备份方案

### Q: Python / Java / Go 三个版本有什么区别？

当前仓库提交的是 Python 版实现。Java / Go 可以作为后续扩展方向：

- **Python版**：当前主实现，适合 AI 工程师、算法工程师、数据工程师方向。
- **Java版规划**：可用 Spring Boot / Spring AI 复刻 API 与服务层，适合 Java 后端方向。
- **Go版规划**：可用 Gin + Neo4j Go Driver + pgvector 客户端实现高并发服务，适合云原生方向。

### Q: 如何演示 DeepResearch 和 Trace Viewer 才更有说服力？

推荐演示顺序：

1. 先上传 1-2 份文档（有实体关系和对比信息）。
2. 用 `/api/qa/deep-research` 提一个复杂问题（对比、评估、风险分析类）。
3. 记录返回的 `run_id`。
4. 打开 `/ui/trace`，按 `run_id` 回放并讲解每一轮：
   - `plan_search`：问题拆解是否合理
   - `retrieve_round`：每轮查询与召回条数
   - `summarize_round`：阶段性结论
   - `assess_gap`：为何继续下一轮或停止
   - `synthesize`：最终整合回答

这样能直观体现你不仅“有答案”，还“有可解释的执行过程”。

### Q: 如何运行测试？

```bash
# Python
cd python
pytest tests/
```

---

## 🔗 参考资料

### 核心框架文档

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/) — Agent编排框架
- [LangChain 官方文档](https://python.langchain.com/docs/get_started/introduction) — LLM应用框架
- [Spring AI 官方文档](https://docs.spring.io/spring-ai/reference/) — Java AI框架
- [Neo4j 官方文档](https://neo4j.com/docs/) — 图数据库
- [ChromaDB 官方文档](https://docs.trychroma.com/) — 向量数据库
- [FastAPI 官方文档](https://fastapi.tiangolo.com/) — Python API框架

### 关键论文

- [RAG原始论文 (2020)](https://arxiv.org/abs/2005.11401) — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- [GraphRAG论文 (2024)](https://arxiv.org/abs/2404.16130) — From Local to Global: A Graph RAG Approach to Query-Focused Summarization
- [Microsoft GraphRAG 开源项目](https://github.com/microsoft/graphrag)

### 相关学习资源

- [LangGraph教程（官方）](https://langchain-ai.github.io/langgraph/tutorials/)
- [Neo4j Graph Academy](https://graphacademy.neo4j.com/) — 免费图数据库课程
- [向量数据库选型对比](https://benchmark.vectorview.ai/vectordbs.html)

---

## 🤝 贡献

欢迎提 Issue 和 PR！

- 发现 Bug？[提交 Issue](https://github.com/goodnight654/DeepResearch-Knowledge-Agents/issues)
- 想加新功能？欢迎 Fork 后提 PR
- 觉得有帮助？请点个 ⭐ Star，这是对我最大的鼓励！

## 🧾 来源与合规说明

本仓库由 [goodnight654](https://github.com/goodnight654) 维护，当前文档、部署说明和 DeepResearch/Trace Viewer 等增强能力围绕本仓库持续整理和演进。

如果本项目基于其他开源项目或早期版本继续开发，公开发布和分发时应保留原项目许可证、版权声明和必要致谢，并在后续补充完整的 `LICENSE` / `NOTICE` 文件。

## 📄 License

当前仓库还没有提交独立的许可证文件。若计划对外分发或用于商业场景，建议先补充与上游许可兼容的 `LICENSE` 文件。
