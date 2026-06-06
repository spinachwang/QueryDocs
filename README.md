# QueryDocs - 年报智能问答系统

> 基于检索增强生成（RAG）技术的公司年报智能问答系统。

## 项目介绍

QueryDocs 是一个基于 RAG（检索增强生成）技术的年报智能问答系统。项目通过结合PDF解析、智能文本分块、混合检索和大语言模型技术，实现对公司年报的智能问答，提供准确、结构化的答案。

### 使用的主要RAG技术

#### 1. PDF解析 (MinerU API)
- 使用MinerU API进行云端PDF解析
- 支持文本、表格、公式、OCR识别
- 输出结构化JSON，保留文档结构

#### 2. 智能文本分块策略
- **类型感知分块**：不同内容类型采用不同策略
  - 原子类型（`title`, `table`, `image`）：不可切断，保证结构完整
  - 可分割类型（`paragraph`, `list`）：按句子边界分割
- **标题+内容组合分块**：标题与后续内容组合，超长时以标题为前缀
- **表格感知分割**：按HTML标签边界（`<tr>`, `<td>`）分割表格，分割时保留表头（`<thead>`）
- 可配置chunk_size（默认300 tokens）和overlap（默认50 tokens）

#### 3. 混合检索
- **向量检索**（FAISS + DashScope/OpenAI嵌入）：捕获语义相似性
- **BM25关键词检索**：数字和专有名词的精确匹配
- **HybridRetriever**：加权融合向量检索和关键词检索分数

#### 4. LLM重排
- **LLMReranker**：使用LLM（Qwen/GPT-4o-mini/MiniMax）评估文档相关性
- **加权融合**：`combined_score = llm_weight * relevance_score + vector_weight * distance`
- 减少幻觉，过滤不相关文档

#### 5. 父文档检索
- 检索时返回完整页面而非小块
- 解决分块导致的上下文碎片化问题
- 适合需要完整段落理解的问题

#### 6. 表格序列化
- 使用LLM将表格HTML转换为结构化信息块
- 包含：核心实体、相关表头、完整描述
- 使表格内容可通过嵌入被检索

#### 7. 结构化输出 + 思维链推理
- Pydantic模型验证输出格式
- 思维链推理：采用链式推理（chain-of-thought reasoning）进行结构化输出
- 不同答案类型使用不同提示模板：
  - `NumberPrompt`：严格指标匹配，拒绝不等价指标
  - `BooleanPrompt`：是/否类答案
  - `NamesPrompt`：实体列表（姓名、职位、产品名）
  - `StringPrompt`：自由文本摘要
  - `ComparativePrompt`：多公司比较结论

#### 8. 多公司比较问题路由
- 检测问题中的多个公司
- 将比较问题分解为单公司子问题
- 并行处理后生成最终比较结论

### 优化方面

#### 性能优化
| 优化项 | 实现方式 |
|--------|----------|
| **并行处理** | ThreadPoolExecutor批量处理问题 |
| **QPS保护** | DashScope设置`max_workers=1`避免限流 |
| **批量嵌入** | 每批25条文档进行向量库构建 |
| **增量保存** | 批量处理中断点保存 |

#### 检索优化
| 优化项 | 实现方式 |
|--------|----------|
| **公司预筛选** | 检索前按公司名过滤 |
| **混合评分** | 向量+关键词+LLM分数加权融合 |
| **相关性校准** | LLM重排提升上下文选择质量 |
| **页码校验** | 验证并过滤幻觉页码 |

#### 答案质量优化
| 优化项 | 实现方式 |
|--------|----------|
| **严格指标匹配** | NumberPrompt拒绝不等价指标 |
| **幻觉检测** | 校验页码引用是否真实存在 |
| **引用验证** | 确保引用的页面在实际上下文中 |
| **增量改进** | 多轮思维链推理 |

### 难点及克服方法

#### 难点1：表格结构保持
**问题**：简单按token数分块会切断表格，破坏其结构。

**解决方法**：类型感知分块 + 表格感知分割：
```python
ATOMIC_TYPES = {'title', 'table', 'image', 'page_header', 'page_number'}
SPLITTABLE_TYPES = {'paragraph', 'list'}

# 按HTML标签边界分割表格
html_parts = re.split(r'(<tr>|</tr>|<td>|</td>)', part)
```

#### 难点2：上下文碎片化
**问题**：小块会丢失段落级上下文，影响综合答案质量。

**解决方法**：父文档检索返回完整页面：
```python
# 返回完整页面而非小块
if return_parent_pages:
    result = {"page": parent_page["page"], "text": parent_page["text"]}
```

#### 难点3：指标幻觉
**问题**：LLM可能返回"相关但不等价"的指标（如营收vs净利润）。

**解决方法**：NumberPrompt中严格指标匹配：
```
**严格指标匹配要求：**
1. 仅当上下文指标含义完全相同时才接受
2. 拒绝情况：范围不匹配、代理指标、需计算推导
3. 对等价性有任何疑问时默认返回'N/A'
```

#### 难点4：页码引用幻觉
**问题**：LLM可能引用不存在的页码。

**解决方法**：页码引用校验：
```python
def _validate_page_references(self, claimed_pages, retrieval_results):
    retrieved_pages = [r['page'] for r in retrieval_results]
    validated = [p for p in claimed_pages if p in retrieved_pages]
    # 过滤幻觉页码，从检索结果补充
```

#### 难点5：多公司比较问题路由
**问题**：比较问题涉及多个公司，单次检索无法处理。

**解决方法**：问题分解管道：
```python
# 1. 检测问题中的公司
# 2. 分解为单公司子问题
# 3. 并行处理
# 4. 生成比较结论
```

### 技术栈

| 类别 | 技术 | 用途 |
|------|------|------|
| PDF解析 | MinerU API | 云端PDF解析 |
| 向量数据库 | FAISS | 相似度检索 |
| 关键词检索 | rank-bm25 | BM25索引 |
| 嵌入模型 | DashScope / OpenAI | 文本向量化 |
| 大语言模型 | Qwen / GPT-4o / MiniMax | 答案生成 |
| API服务 | FastAPI | RESTful接口 |
| 并发处理 | concurrent.futures | 并行处理 |
| 结构验证 | Pydantic | 输出格式验证 |

### 快速开始

#### 1. 安装与配置

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API密钥（将 env 文件改名为 .env 并填入真实 key）
cp env .env
# 必需: MINERU_API_KEY（PDF解析）
# 按需: DASHSCOPE_API_KEY / OPENAI_API_KEY / MINIMAX_API_KEY（推理与嵌入）
# 可选: JINA_API_KEY / GEMINI_API_KEY（重排/多模型）
```

#### 2. 离线构建索引

CLI 入口 (`main.py`) 按以下顺序串联整个流水线：

```bash
# 1) 解析 PDF（云端 MinerU 解析 → content_list_v2.json）
python main.py parse-pdfs --parallel --max-workers 10

# 2) （可选）表格序列化：把表格 HTML 转成可被检索的结构化信息块
python main.py serialize-tables --max-workers 10

# 3) 报告分块：按类型感知策略切分到 chunked_reports/
python main.py process-reports --config no_ser_tab   # 不含表格序列化
python main.py process-reports --config ser_tab      # 含表格序列化

# 4) 构建向量库（FAISS）和 BM25 索引（按需）
python main.py build-vectors
python main.py build-bm25
```

> **配置字典说明**：`process-reports` 用的是 [`preprocess_configs`](src/pipeline.py#L188-L189)，只有 `ser_tab / no_ser_tab` 两个选项；`process-questions` 用的是 [`configs`](src/pipeline.py#L220-L235)，见下方"运行配置"小节。

#### 3. 在线问答（CLI 批量）

```bash
python main.py process-questions --config minimax
# 可选 --config: base / pdr / max / minimax
```

#### 4. 启动 FastAPI 服务

```bash
# 方式 A：在项目根目录运行（推荐）
uvicorn src.api.main:app --reload --port 8000

# 方式 B：进入 src/api 运行
cd src/api
uvicorn main:app --reload --port 8000
```

#### 5. 启动前端（可选）

```bash
cd frontend
npm install
npm run dev
```

### 项目结构

```
QueryDocs/
├── main.py                       # CLI 入口（Click 7 个子命令：parse-pdfs/serialize-tables/process-reports/build-vectors/build-bm25/process-questions/evaluate）
├── requirements.txt
├── env                           # 环境变量模板（需改名 .env）
├── questions.json                # 待回答问题列表
├── subset.csv                    # 公司清单（sha1, file_name, company_name）
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py               # 主管道：RunConfig / PipelineConfig / Pipeline
│   ├── pdf_mineru.py             # PDF 解析（MinerU 云端 API）
│   ├── parsed_reports_merging.py # 解析结果规整为页文本
│   ├── text_splitter.py          # 类型感知智能分块
│   ├── process_chunks.py         # 批量分块（跳过 _p1-200 等拆分目录）
│   ├── ingestion.py              # FAISS 向量库 + BM25 索引构建
│   ├── retrieval.py              # VectorRetriever / BM25Retriever / HybridRetriever
│   ├── reranking.py              # LLM/Jina 重排
│   ├── questions_processing.py   # 单/多公司问答主逻辑
│   ├── tables_serialization.py   # TableSerializer（LLM 表格 → 结构化信息块）
│   ├── prompts.py                # 提示词 + Pydantic Schema
│   ├── api_requests.py           # OpenAI/DashScope/MiniMax 多 API 统一封装
│   ├── api_request_parallel_processor.py  # 并发限流批处理
│   ├── evaluation/                # RAGAS 离线评估
│   │   ├── __init__.py
│   │   ├── ragas_judge.py         # 包装 MiniMax(DashScope) judge/embedder
│   │   └── evaluate.py            # CLI 评估脚本（load_questions/build_samples/run_ragas/save_report）
│   └── api/                      # FastAPI 服务
│       ├── __init__.py
│       ├── main.py               # FastAPI app 入口
│       ├── models.py             # Pydantic 请求/响应模型
│       ├── pipeline_wrapper.py   # 惰性单例 + Pipeline 初始化
│       └── routers/
│           ├── __init__.py
│           └── qa.py             # POST /api/qa/ask 路由
│
├── tests/                        # pytest 单测（conftest + evaluate + ragas_judge）
│
├── data/stock_data/              # 数据目录
│   ├── pdf_reports/              # 原始 PDF
│   ├── questions.json            # 问题列表（同根目录副本）
│   ├── subset.csv                # 公司清单（同根目录副本）
│   ├── debug_data/
│   │   ├── 03_reports_markdown/      # MinerU 解析结果（content_list_v2.json + full.md）
│   │   ├── 03_reports_markdown_ser_tab/  # 表格序列化版本
│   │   ├── chunked_reports/          # 分块结果
│   │   └── databases/
│   │       ├── vector_dbs/           # FAISS 向量库
│   │       └── bm25_dbs/             # BM25 pickle
│   └── answers_*.json            # 批量问答结果（按 config_suffix 自动编号）
│
├── frontend/                     # React + TypeScript + Vite 前端
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
│
└── docs/
    ├── architecture.md           # 架构文档
    └── src_modules_overview.md   # src 模块速查
```

### CLI 子命令一览

| 命令 | 配置参数 | 作用 |
|------|----------|------|
| `parse-pdfs` | `--parallel / --sequential`、`--max-workers` | 批量 MinerU 解析 PDF |
| `serialize-tables` | `--max-workers` | LLM 表格序列化（可选步骤） |
| `process-reports` | `--config {ser_tab,no_ser_tab}` | 报告分块到 `chunked_reports/` |
| `build-vectors` | （无） | 构建 FAISS 向量库到 `vector_dbs/` |
| `build-bm25` | （无） | 构建 BM25 索引到 `bm25_dbs/` |
| `process-questions` | `--config {base,pdr,max,minimax}` | 批量问答并输出 `answers_*.json` |
| `evaluate` | `--questions` / `--output` / `--limit` / `--skip-judge` / `--skip-embedder` | RAGAS 离线评估，输出 `report.csv` + `summary.md` |

### 运行配置（`process-questions --config`）

`configs` 字典在 [src/pipeline.py](src/pipeline.py#L220-L235) 定义：

| 配置 | 主要特性 | LLM |
|------|---------|-----|
| `base` | 基础：向量检索 + 路由 + 结构化 CoT | GPT-4o-mini |
| `pdr` | 父文档检索 | GPT-4o |
| `max` | LLM 重排 + 并行 | Qwen-Turbo |
| `minimax` | LLM 重排 + 自定义路径 | MiniMax-M2.7 |

### API 接口

服务启动后默认监听 `http://localhost:8000`，交互文档见 `/docs`。

| Method | Path | 说明 |
|--------|------|------|
| GET | `/` | 服务信息 |
| GET | `/api/health` | 健康检查 |
| POST | `/api/qa/ask` | 单问题推理（CoT 答案 + 引用） |

**请求** `POST /api/qa/ask`：

```json
{
  "question": "请简要总结某公司2022年主营业务内容。",
  "kind": "string"
}
```

`kind` 可选：`string` / `number` / `boolean` / `names`，默认 `string`。

**响应**：

```json
{
  "step_by_step_analysis": "分步推理（≥5 步 150 字）",
  "reasoning_summary": "推理摘要",
  "relevant_pages": [1, 2, 3],
  "final_answer": "最终答案或 'N/A'",
  "references": [
    { "pdf_sha1": "stock_10001", "page_index": 1 }
  ],
  "contexts": [
    "检索到的 chunk 原文（最多 10 条，每条 ≤ 2000 字符）",
    "..."
  ]
}
```

> `references` 字段由后端从检索结果中提取页码引用，与 `relevant_pages` 互为补充。`contexts` 字段返回检索到的切片原文，主要供 RAGAS 离线评估使用，前端一般不需要展示。

### 离线评估（RAGAS）

RAGAS 接入 [`src/evaluation/`](src/evaluation/)，用于量化比较切块/重排/表格序列化等改动带来的检索/生成质量变化。

#### 评估指标（无监督版，不需金标样）

| 指标 | 衡量 |
|---|---|
| `faithfulness` | 答案是否忠于检索到的 contexts |
| `answer_relevancy` | 答案是否切题（需 embedder） |
| `llm_context_precision_without_reference` | 检索的 chunks 是否相关 |
| `nv_context_relevance` | 检索的 chunks 冗余度 |
| `nv_response_groundedness` | 答案多大程度由检索内容支撑 |

#### 模型分工

| 角色 | 提供方 | 用途 |
|---|---|---|
| **Judge LLM** | MiniMax-M2.7（Anthropic 兼容 API，max_tokens=8192） | faithfulness / context_precision 等需要 LLM 打分的指标 |
| **Embedder** | DashScope text-embedding-v4 | answer_relevancy 的反向问题嵌入比对 |

环境变量：`MINIMAX_API_KEY` + `DASHSCOPE_API_KEY`。

#### 用法

```bash
# 跑完整评估（默认读 questions.json，写到 data/eval_results/{ts}/）
python main.py evaluate

# 试跑 3 条
python main.py evaluate --limit 3

# 仅收集样本不调 judge（不花 token，验证 pipeline 能产出 contexts）
python main.py evaluate --skip-judge --output data/eval_results/smoke

# 自定义问题集和输出目录
python main.py evaluate --questions path/to/q.json --output data/eval_results/exp1
```

输出：
- `data/eval_results/{ts}/report.csv`：每题各指标分数
- `data/eval_results/{ts}/summary.md`：聚合均值 + 运行时长

#### A/B 对比工作流

1. 跑基线：`python main.py evaluate --output data/eval_results/baseline`
2. 改配置 / 切块 / 重排参数
3. 跑改动后：`python main.py evaluate --output data/eval_results/v2`
4. 对比 `summary.md` 里的均值差异

#### 测试

```bash
PYTHONIOENCODING=utf-8 conda run -n rag-cy pytest tests/ --cov=src.evaluation --cov-report=term
```

当前覆盖率 ~90%（24 个用例）。

### 许可证

MIT