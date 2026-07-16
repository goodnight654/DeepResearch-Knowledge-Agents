# DeepResearch 完整教程

本文对应当前仓库的 Python 实现，覆盖环境搭建、配置、调用、证据引用、Trace 回放、测试、故障排查和二次开发。所有命令默认从仓库根目录执行。

## 1. DeepResearch 解决什么问题

普通 QA 适合一次检索即可回答的问题。DeepResearch 面向对比、评估、调研和风险分析等开放问题，使用以下循环：

```text
plan_search
    ↓
retrieve_round → summarize_round → assess_gap
    ↑                                  │
    └──────── 证据仍不足，生成追问 ──────┘
                                       ↓
                                  synthesize
```

每轮都会记录查询、召回数量、阶段总结、信息缺口和置信度。最终回答中的 `[E1]`、`[E2]` 与响应里的 `evidence` 一一对应。

当前实现额外保证：

- 向量库、知识图谱和 Web 检索并行执行，单个后端故障不会中断整轮研究；
- 跨轮去除重复查询，避免模型反复搜索同一句话；
- 默认至少收集 2 条证据和 2 个独立来源，数量可配置；
- 优先选择不同来源的上下文，再按相关性补足；
- 删除模型生成但实际不存在的证据编号，并返回 `citation_warnings`；
- 置信度同时参考模型判断、证据质量、来源多样性和引用覆盖率；无证据时固定为 0；
- 网页读取拒绝本机、私网地址和跳转到私网的 URL，降低 SSRF 风险。

## 2. 安装

推荐 Python 3.10 或更高版本：

```bash
python3 -m venv python/.venv
source python/.venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r python/requirements.txt
```

复制配置文件：

```bash
cp python/.env.example python/.env
```

应用会从当前工作目录读取 `.env`。推荐进入 `python/` 后启动：

```bash
cd python
python -m api.main
```

访问以下地址验证：

- 健康检查：`http://localhost:8080/api/health`
- Swagger：`http://localhost:8080/docs`
- Trace Viewer：`http://localhost:8080/ui/trace`

没有 API Key 或本地数据库时，服务仍可启动并返回健康状态；需要模型生成或向量嵌入时才必须提供对应配置。

## 3. 核心配置

最小可用配置：

```env
OPENAI_API_KEY=sk-your-key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

WEB_SEARCH_PROVIDER=duckduckgo
WEB_PAGE_FETCH_ENABLED=true
```

DeepResearch 调优项：

```env
# 最多研究轮数
DEEPRESEARCH_MAX_ITERATIONS=3

# 每轮最多并行执行多少个查询
DEEPRESEARCH_QUERIES_PER_ITERATION=3

# 每个查询保留多少条上下文
DEEPRESEARCH_CONTEXTS_PER_QUERY=4

# 最终交给综合节点的上下文上限
DEEPRESEARCH_FINAL_CONTEXT_LIMIT=12

# 提前结束前建议满足的证据数量和独立来源数
DEEPRESEARCH_MIN_EVIDENCE=2
DEEPRESEARCH_MIN_SOURCES=2
```

Web 配置：

```env
WEB_SEARCH_PROVIDER=duckduckgo  # disabled | duckduckgo | tavily | serpapi
WEB_SEARCH_TOP_K=5
WEB_SEARCH_TIMEOUT_SECONDS=10

TAVILY_API_KEY=
SERPAPI_API_KEY=

WEB_PAGE_FETCH_ENABLED=true
WEB_PAGE_TIMEOUT_SECONDS=10
WEB_PAGE_MAX_CHARS=12000
WEB_PAGE_CHUNKS_PER_RESULT=2
WEB_PAGE_BLOCK_PRIVATE_NETWORKS=true
```

生产环境不要关闭 `WEB_PAGE_BLOCK_PRIVATE_NETWORKS`。只有在完全可信、隔离的测试网络中读取内网页面时才考虑设为 `false`。

本地知识库配置：

```env
VECTOR_STORE_TYPE=chroma
CHROMA_HOST=localhost
CHROMA_PORT=8000

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your-password

# 外部依赖启动最多等待几秒，超时后进入降级模式
DEPENDENCY_INIT_TIMEOUT_SECONDS=3
```

## 4. 发起研究请求

```bash
curl -sS -X POST http://localhost:8080/api/qa/deep-research \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "比较向量 RAG 与 GraphRAG 在准确性、成本和维护复杂度上的差异，并说明适用边界"
  }'
```

典型响应结构：

```json
{
  "run_id": "deepresearch_20260716T010203Z_ab12cd34ef",
  "question": "...",
  "executive_summary": "...",
  "answer": "结论一 [E1]；结论二 [E2]。",
  "confidence": 0.82,
  "iterations": 2,
  "steps": [
    {
      "iteration": 1,
      "focus": "...",
      "queries": ["..."],
      "summary": "...",
      "gaps": ["缺少成本数据"],
      "contexts_count": 4
    }
  ],
  "evidence": [
    {
      "evidence_id": "E1",
      "source": "https://example.com/article",
      "title": "Example article",
      "quote": "用于核验结论的正文片段",
      "confidence": 0.91,
      "retrieved_at": "2026-07-16T01:02:03+00:00"
    }
  ],
  "cited_evidence_ids": ["E1", "E2"],
  "citation_warnings": [],
  "sources": []
}
```

兼容旧客户端的 `/api/qa/deep-search` 仍然可用，但新代码应使用 `/api/qa/deep-research`。

## 5. 如何理解证据和置信度

`evidence` 是最终被选中的证据表，不等同于所有搜索结果。系统先按内容和来源去重，然后优先保留不同来源。

引用处理规则：

1. 只有 `evidence` 中存在的编号才算有效；
2. 模型返回 `[E99]` 但证据表没有 E99 时，该编号会被删除并写入 `citation_warnings`；
3. 有证据但模型没有引用时，系统会附加证据索引并给出警告；
4. 没有证据时，置信度固定为 0，回答会明确提示证据不足。

置信度不是事实正确率。它是用于排序和告警的启发式值，不能代替人工核验。高风险结论应打开原始 `source` 检查上下文。

## 6. 查看和回放 Trace

请求成功后记录 `run_id`，浏览器打开：

```text
http://localhost:8080/ui/trace?run_id=你的run_id
```

也可以直接读取 JSON：

```bash
curl -sS http://localhost:8080/api/qa/runs/你的run_id
curl -sS 'http://localhost:8080/api/qa/runs?limit=20&workflow=deepresearch'
```

主要事件：

| 事件 | 重点检查 |
|---|---|
| `plan` | 目标、子问题、第一轮查询是否合理 |
| `retrieve_round` | 查询是否重复、召回数量是否异常 |
| `summarize_round` | 总结是否只使用现有材料 |
| `assess_gap` | 为何继续或停止、还缺什么证据 |
| `synthesize` | 证据数、有效引用、最终置信度 |

失败的 DeepResearch 也会保存一条 `status=failed` 的记录，接口错误信息中会返回对应 `run_id`。

## 7. 离线和降级行为

系统允许三个检索通道独立失败：

- Chroma/PGVector 不可用：跳过向量检索；
- Neo4j 不可用：跳过实体、邻居和 Cypher 检索；
- Web Provider 不可用：继续使用本地知识库；
- LLM 未配置：使用原问题作为搜索计划，并根据检索证据生成确定性的降级摘要。

健康检查示例：

```json
{
  "status": "ok",
  "components": {
    "vector_store": "degraded",
    "knowledge_graph": "degraded",
    "web_search": "ready",
    "llm_configured": false
  }
}
```

`status=ok` 表示 API 本身可服务；具体能力以 `components` 为准。

## 8. 运行测试和静态检查

从仓库根目录执行：

```bash
python/.venv/bin/python -m pytest
python/.venv/bin/ruff check python
python/.venv/bin/python -m compileall -q python
```

测试使用真实的 Pydantic、FastAPI 和 LangGraph，不会再用自制模块桩替代框架。测试覆盖：

- 四条 LangGraph 工作流的状态传递和分支合并；
- DeepResearch 多轮循环、查询去重、来源多样性和引用校验；
- 向量库或图数据库不可用时的降级；
- 网页正文提取、分块、私网 URL 拦截和搜索结果回退；
- API 启动、请求校验、失败 trace、上传文件名和大小限制；
- Trace 原子写入、读取和过滤。

## 9. 常见问题

### 服务启动了，但回答没有模型生成内容

检查健康接口里的 `llm_configured`，再确认 `.env` 位于启动命令的当前目录。若从仓库根目录启动，可以显式导出环境变量，或先 `cd python`。

### DeepResearch 只有 Web 证据，没有本地文档

检查 `vector_store` 与 `knowledge_graph` 是否为 `ready`，并确认文档已经通过 `/api/ingest/upload` 入库。数据库未初始化时，读取会安全返回空结果。

### DuckDuckGo 没有结果

公开搜索端点可能限流。稳定环境推荐配置 Tavily 或 SerpAPI；离线环境设为 `disabled`。

### 为什么研究跑满最大轮数

通常是未达到 `DEEPRESEARCH_MIN_EVIDENCE` 或 `DEEPRESEARCH_MIN_SOURCES`。查看 `assess_gap` 事件的 `gaps` 和 `follow_up_queries`，必要时降低门槛或改善搜索 Provider。

### 为什么引用被删除

模型输出了证据表中不存在的编号。查看响应的 `citation_warnings` 和 `evidence`，不要手工信任被删除的引用。

## 10. 二次开发入口

- 对外 DeepResearch 导入：`python/agents/deepresearch_agent.py`
- 研究规划、缺口分析、证据与引用实现：`python/agents/deepsearch_agent.py`（保留旧导入兼容）
- LangGraph 多轮编排：`python/orchestrator/graph.py`
- 向量/图/Web 统一检索：`python/services/hybrid_retriever.py`
- 搜索 Provider：`python/services/web_search.py`
- 网页安全读取：`python/services/web_page_reader.py`
- API 与响应模型：`python/api/main.py`
- 运行轨迹：`python/services/run_trace_store.py`

新增搜索 Provider 时，实现与 `WebSearchService.search()` 相同的返回结构；新增证据类型时，统一转换为 `RetrievedContext`，这样去重、重排、引用和 Trace 无需重复实现。
