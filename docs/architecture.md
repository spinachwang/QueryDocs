# QueryDocs 架构文档

## 1. 项目概述

**项目名称**: QueryDocs - 年报智能问答系统
**核心功能**: 对公司年报PDF进行深度解析，构建RAG管道，实现基于检索增强生成的智能问答

### 技术特点

- **PDF解析**: MinerU API 云端批量解析，支持文本、表格、公式、OCR
- **智能分块**: 按内容类型（原子类型/可分割类型）分别处理，保证重要结构不被切断
- **混合检索**: FAISS 向量数据库 + BM25 关键词检索 + LLM 重排
- **结构化输出**: Pydantic 模型验证输出格式，Chain-of-Thought 推理
- **多API支持**: OpenAI / DashScope (Qwen) / MiniMax / Google Gemini

---

## 2. 系统架构图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              用户交互层                                       │
│  ┌─────────────────────────────┐         ┌──────────────────────────────┐   │
│  │   main.py                   │         │   src/api/main.py            │   │
│  │   (CLI 入口: 6 个子命令)     │         │   (FastAPI 服务)              │   │
│  └──────────────┬──────────────┘         └──────────────┬───────────────┘   │
│                 │                                       │                   │
└─────────────────┼───────────────────────────────────────┼───────────────────┘
                  │                                       │
                  ▼                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             管道编排层 (Pipeline)                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │  src/pipeline.py                                                         ││
│  │   • Pipeline.__init__()        - 加载路径与 RunConfig                      ││
│  │   • process_questions()        - 批量问题处理 → answers_*.json            ││
│  │   • answer_single_question()  - 单问题推理（供 API 调用）                 ││
│  │                                                                          ││
│  │  ⚠️ Pipeline 类本身只做"在线推理"编排；离线索引（PDF 解析、分块、         ││
│  │  向量化、BM25 索引、表格序列化）由 CLI 子命令直接驱动对应模块，            ││
│  │  不走 Pipeline 类。                                                        ││
│  └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             核心处理层                                        │
│                                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │ pdf_mineru   │  │text_splitter │  │   prompts    │  │tables_serial │  │
│  │              │  │              │  │              │  │               │  │
│  │• MinerU API  │  │• 类型感知 │  │• CoT推理 │  │• LLM表格结构化│  │
│  │  云端批量解析 │  │  智能分块    │  │• 结构化输出  │  │• 异步批处理   │  │
│  │ │  │• 标题+内容   │  │• 多类型答案  │  │               │  │
│  │              │  │  组合分块    │  │  分类处理 │  │               │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  └───────────────┘  │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         ingestion.py                                 │ │
│  │  ┌─────────────────────┐    ┌─────────────────────────────────────┐ │ │
│  │  │   VectorDBIngestor  │    │          BM25Ingestor                 │ │ │
│  │  │  • DashScope嵌入   │    │  • BM25Okapi索引构建 │ │ │
│  │  │  • FAISS IndexFlatIP│    │  • pickle序列化                      │ │ │
│  │  │  • 批量处理+重试   │    │  • 按公司独立索引                   │ │ │
│  │  └─────────────────────┘    └─────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         retrieval.py                                 │ │
│  │  ┌───────────────┐ ┌───────────────┐ ┌───────────────────────────┐│ │
│  │  │VectorRetriever│ │BM25Retriever │ │    HybridRetriever         ││ │
│  │  │• FAISS搜索 │ │• 关键词匹配   │ │• 向量+BM25融合 ││ │
│  │  │• 按公司筛选 │ │• 按公司筛选   │ │• LLM重排融合              ││ │
│  │  │• 父文档检索  │ │               │ │• 加权分数计算             ││ │
│  │  └───────────────┘ └───────────────┘└───────────────────────────┘│ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                         reranking.py                                 │ │
│  │  ┌─────────────────┐         ┌─────────────────────────────────────┐│ │
│  │  │   JinaReranker  │         │         LLMReranker ││ │
│  │  │  (外部API)       │         │  • DashScope/OpenAI/MiniMax LLM      ││ │
│  │  │                 │         │  • 单块/批量重排 ││ │
│  │  │                 │         │  • 多线程并行处理                      ││ │
│  │  └─────────────────┘         │  • QPS限制保护                        ││ │
│  │                             └─────────────────────────────────────┘ │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                     questions_processing.py │ │
│  │  • 公司名自动提取 (从subset匹配)                                      │ │
│  │  • 单公司问答 / 多公司比较问答                                        │ │
│  │  • 并行请求处理 +断点保存                                            │ │
│  │  • 页码引用校验与修复 │ │
│  │  • LLM幻觉检测与纠正 │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                        api_requests.py │ │
│  │  ┌────────────────┐ ┌────────────────┐ ┌────────────────┐          │ │
│  │  │BaseOpenAIProc. │ │BaseDashscope │ │BaseMiniMaxProc │          │ │
│  │  │               │ │ Processor │ │               │          │ │
│  │  └────────────────┘ └────────────────┘ └────────────────┘          │ │
│  │ │                                    │ │
│  │                    ┌──────────┴──────────┐                        │ │
│  │                    │    APIProcessor    │                        │ │
│  │                    │  (统一调度入口)    │                        │ │
│  │                    └─────────────────────┘                        │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ process_chunks.py │ │
│  │  • 批量处理年报分块                                                   │ │
│  │  •跳过拆分目录（_p1-200, _p201-222等）                              │ │
│  │  • 按PDF文件名聚合chunks                                             │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
            │
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              数据持久层                                      │
│  ┌────────────────────┐   ┌────────────────────┐   ┌────────────────────┐  │
│  │   PDF Reports      │   │   Vector DB (FAISS) │   │   BM25 DB (pickle) │  │
│  │   年报PDF文件       │   │   向量索引文件      │   │   BM25索引文件     │  │
│  └────────────────────┘   └────────────────────┘   └────────────────────┘  │
│                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐  │
│  │              Chunked Reports (JSON)                                    │  │
│  │  { metainfo: {source, total_chunks},                                  │  │
│  │    content: { chunks: [{id, page, type, text, length_tokens}] } }      │  │
│  └─────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. 核心模块详解

### 3.1 入口层

| 文件 | 说明 |
|------|------|
| `main.py` | CLI 命令行入口（Click），提供 6 个子命令：`parse-pdfs` / `serialize-tables` / `process-reports` / `build-vectors` / `build-bm25` / `process-questions` |
| `src/api/main.py` | FastAPI Web 服务入口，注册 `qa` 路由与 `/api/health`、`/` |
| `src/api/routers/qa.py` | `POST /api/qa/ask` 单问题推理路由 |
| `src/api/pipeline_wrapper.py` | 惰性单例的 Pipeline 包装，API 启动时按 `minimax` 配置初始化一次 |
| `frontend/` | React + TypeScript + Vite 前端（独立子项目） |

### 3.2 管道编排 (Pipeline)

**文件**: `src/pipeline.py`

`Pipeline` 类主要承担**在线推理编排**，负责把检索 + 重排 + LLM 推理串起来：

```
用户问题 → 公司匹配 → 检索 → (重排) → LLM 推理 → 页码校验 → 结构化答案
```

**关键配置**:
- `RunConfig`: 运行时参数配置（是否使用序列化表格、父文档检索、LLM重排、并发数等）
- `PipelineConfig`: 路径配置（数据目录、输出目录、数据库路径）

**实际方法**:
- `process_questions()`: 批量读取 `questions.json`，输出 `answers{config_suffix}.json`（同名文件自动加编号后缀）
- `answer_single_question(question, kind)`: 单问题即时推理，供 FastAPI 路由调用

> **离线索引（PDF 解析、分块、向量化、BM25 索引、表格序列化）由 `main.py` 的子命令直接驱动对应模块**，不通过 `Pipeline` 类。`Pipeline` 类里也**没有** `parse_pdf_reports / chunk_reports / create_vector_dbs / create_bm25_db` 这类方法。

### 3.3 PDF解析

**文件**: `src/pdf_mineru.py`

使用 **MinerU API** 进行PDF解析：

```
MinerUAPI
├── batch_get_upload_urls()   - 获取批量上传URL
├── upload_file()             - 上传文件到预签名URL
├── get_batch_results()        - 获取批量任务结果
├── wait_for_completion()      - 轮询等待任务完成
└── download_and_extract()     - 下载并解压结果

MinerUParser
└── parse()                   - 批量解析PDF文件
```

**技术细节**:
- 调用 `https://mineru.net/api/v4/file-urls/batch` 获取上传URL
- 使用预签名URL直接上传文件
- 调用 `https://mineru.net/api/v4/extract-results/batch/{batch_id}` 查询状态
- 支持状态: `pending` → `running` → `converting` → `done`
- 输出: `{data_id}/content_list_v2.json` 结构化内容

### 3.4 文本分块 (智能类型感知分块)

**文件**: `src/text_splitter.py`

`TextSplitter` 类实现了**类型感知的智能分块策略**：

#### 内容类型定义

```python
# 原子类型：不可切断，保证结构完整性
ATOMIC_TYPES = {'title', 'table', 'image', 'page_header', 'page_number'}

# 可分割类型：可以按自然边界切断
SPLITTABLE_TYPES = {'paragraph', 'list'}
```

#### 分块策略

| 类型 | 处理方式 |
|------|----------|
| `title` | 标题不单独成chunk，与下一个内容组合；超长标题作为前缀保留 |
| `paragraph` | 按句子分割，支持chunk_size和overlap |
| `table` | 按HTML标签分割(`<tr>`, `<td>`)，超长分割时保留表头(`<thead>`) |
| `list` | 作为整体保留，或按句子边界分割 |
| `image` | 保留图片路径引用 |

#### 关键算法

```python
def split_content_list_v2(content_list_v2_path, output_path,
                           chunk_size=300, overlap=50, pdf_name=None):
    # 1. 按页遍历MinerU的content_list_v2.json
    # 2. 遇到标题暂存，与后续内容组合
    # 3. 超长内容按句子分割，保持语义完整
    # 4. 表格按HTML标签边界分割，超长时保留表头(<thead>)
    # 5. 输出JSON格式的chunks
```

**输出格式** (JSON):
```json
{
  "metainfo": { "source": "中芯国际2024年年度报告.pdf", "total_chunks": 150 },
  "content": {
    "chunks": [
      { "id": 0, "page": 1, "type": "content", "text": "...", "length_tokens": 280 },
      { "id": 1, "page": 2, "type": "content", "text": "...", "length_tokens": 320 }
    ]
  }
}
```

### 3.5 索引构建 (Ingestion)

**文件**: `src/ingestion.py`

#### VectorDBIngestor
- 使用 **DashScope TextEmbedding** 或 **OpenAI Embedding** 获取文本向量
- 调用 **FAISS IndexFlatIP** 构建向量库（内积相似度，等价于余弦相似度）
- 批量处理报告，每批25条嵌入
- 重试机制: 20秒等待，最多2次

#### BM25Ingestor
- 使用 `rank_bm25.BM25Okapi` 构建BM25索引
- 每个报告独立索引，文件名用 `doc_{md5(source)}`格式
- pickle序列化存储

### 3.6 检索 (Retrieval)

**文件**: `src/retrieval.py`

#### VectorRetriever
- 按公司名检索对应年报
- 支持 DashScope / OpenAI 嵌入
- 返回top_n个最相似文本块
- 支持**父文档检索**（返回完整页面而非分块）

#### BM25Retriever
- 传统关键词检索
- 按公司名筛选文档

#### HybridRetriever
- **混合检索核心组件**
- 首轮向量检索获取候选
- 调用 `LLMReranker` 进行重排
- 加权融合: `combined_score = llm_weight * relevance_score + vector_weight * distance`

### 3.7 LLM重排 (Reranking)

**文件**: `src/reranking.py`

#### LLMReranker
- 支持 DashScope (Qwen) / OpenAI / MiniMax
- **单块重排**: `get_rank_for_single_block()`
- **批量重排**: `get_rank_for_multiple_blocks()`
- 多线程并行，`max_workers=1` 保证DashScope QPS限制
- 相关性评分 (0-1)，与向量距离加权融合

#### JinaReranker
- 基于 Jina API 的外部重排服务（多语言支持）

### 3.8 问题处理 (Questions Processing)

**文件**: `src/questions_processing.py`

`QuestionsProcessor` 是RAG流程的核心编排类：

```
用户问题
    │
    ▼
_extract_companies_from_subset()  ── 从问题中匹配公司名
    │
    ├── [单公司] ──→ get_answer_for_company()
    │                    │
    │                    ▼
    │              检索 (VectorRetriever / HybridRetriever)
    │                    │
    │                    ▼
    │格式化 RAG Context
    │                    │
    │                    ▼
    │              APIProcessor.get_answer_from_rag_context()
    │                    │
    │                    ▼
    │              页码引用校验 → LLM幻觉检测 → 结构化答案
    │
    └── [多公司] ──→ process_comparative_question()
                       │
                       ▼
                 分解为单公司问题并行处理
                       │
                       ▼
                 汇总答案 → 调用LLM生成比较结论
```

**关键方法**:

| 方法 | 说明 |
|------|------|
| `_extract_companies_from_subset()` | 从问题文本匹配subset.csv中的公司 |
| `get_answer_for_company()` | 单公司RAG问答 |
| `process_comparative_question()` | 多公司比较问答 |
| `_validate_page_references()` | 校验LLM引用的页码是否存在，过滤幻觉 |
| `_post_process_submission_answers()` | 提交格式转换 (1-based → 0-based) |

### 3.9 API请求处理

**文件**: `src/api_requests.py`

```
                    ┌────────────────────────┐
                    │    APIProcessor        │
                    │    (统一入口)          │
                    └───────────┬────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        │                       │                       │
        ▼                       ▼                       ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│BaseDashscope  │     │ BaseOpenAIProc │     │ BaseMiniMaxProc│
│ Processor     │     │               │     │               │
└───────────────┘     └───────────────┘     └───────────────┘
        │                       │                       │
       └───────────────────────┼───────────────────────┘
                                ▼
                    ┌────────────────────────┐
                    │ get_answer_from_rag_   │
                    │ context()              │
                    │ • 构建RAG Prompt       │
                    │ • 结构化输出 (Pydantic) │
                    │ • 统一返回格式          │
                   └────────────────────────┘
```

### 3.10 提示词工程 (Prompts)

**文件**: `src/prompts.py`

| 提示模板类 | 用途 |
|-----------|------|
| `AnswerWithRAGContextSharedPrompt` | RAG问答基础模板 |
| `AnswerWithRAGContextStringPrompt` | 字符串类答案 |
| `AnswerWithRAGContextNumberPrompt` | 数值类答案（含严格指标匹配） |
| `AnswerWithRAGContextBooleanPrompt` | 是/否类答案 |
| `AnswerWithRAGContextNamesPrompt` | 名单/实体列表答案 |
| `ComparativeAnswerPrompt` | 多公司比较结论 |
| `RephrasedQuestionsPrompt` | 比较问题分解为单公司问题 |
| `RerankingPrompt` | LLM重排评分 |

**答案结构** (Pydantic):
```python
{
  "step_by_step_analysis": "详细分步推理，至少5步150字",
  "reasoning_summary": "推理摘要，约50字",
  "relevant_pages": [页码列表],
  "final_answer": "最终答案或'N/A'"
}
```

### 3.11 表格序列化

**文件**: `src/tables_serialization.py`

`TableSerializer` 使用LLM将表格HTML转换为结构化信息块：

```
表格HTML + 上下文 → LLM → 信息块列表
                          │
                          ▼
                 SerializedInformationBlock
                 ├── subject_core_entity: 核心实体
                 ├── relevant_headers_list: 相关表头
                 └── information_block: 完整描述
```

- 支持同步/异步处理
- 异步模式使用 `AsyncOpenaiProcessor` 并发请求
- 多线程池处理文件级别并行

### 3.12 批量分块处理

**文件**: `src/process_chunks.py`

`process_all_reports()` 函数批量处理年报分块：

-遍历 `03_reports_markdown` 目录下的年报
- **跳过拆分目录**（如 `_p1-200`, `_p201-222`），避免重复处理
- 查找 `*_content_list_v2.json` 文件
- 调用 `TextSplitter.split_content_list_v2()` 进行分块
- 输出到 `databases/chunked_reports/`

---

## 4. 数据流

### 4.1 离线索引构建流程

```
1. PDF文件 (pdf_reports/)
       │
       ▼
2. MinerUParser.parse()  [src/pdf_mineru.py]
       │  • 云端MinerU解析
       │  • 输出content_list_v2.json
       │
       ▼
3. Parsed JSON (debug_data/03_reports_markdown/)
       │
       ▼
4. process_chunks.py  [src/process_chunks.py]
       │  • 批量分块处理
       │  • 跳过拆分目录
       │
       ▼
5. Chunked Reports (databases/chunked_reports/)
       │
       ├──▶ VectorDBIngestor.process_reports()  [src/ingestion.py]
       │         • FAISS向量库
       │         • DashScope/OpenAI嵌入
       │         ▼
       │    Vector DBs (databases/vector_dbs/*.faiss)
       │
       └──▶ BM25Ingestor.process_reports()  [src/ingestion.py]
                 • BM25索引
                 • pickle序列化
                 ▼
            BM25 DBs (databases/bm25_dbs/*.pkl)
```

### 4.2 在线推理流程

```
用户问题
    │
    ▼
QuestionsProcessor.process_single_question()
    │
    ├─▶ _extract_companies_from_subset()  匹配公司
    │
    ├─▶ [单公司]
    │    VectorRetriever / HybridRetriever
    │ │
    │         ▼
    │    LLM重排 (可选)
    │         │
    │         ▼
    │    APIProcessor.get_answer_from_rag_context()
    │         │
    │         ▼
    │    页码校验 + LLM幻觉检测 + 结构化答案
    │
    └─▶ [多公司比较]
         process_comparative_question()
              │
              ├─▶ 并行处理各公司子问题
              │
              └─▶ ComparativeAnswerPrompt
                       │
                       ▼
                  比较结论
```

---

## 5. 配置系统

### 5.1 运行时配置 (RunConfig)

```python
@dataclass
class RunConfig:
    use_serialized_tables: bool = False      # 是否使用序列化表格
    parent_document_retrieval: bool = False   # 父文档检索
    use_vector_dbs: bool = True              # 使用向量检索
    use_bm25_db: bool = False # 使用BM25
    llm_reranking: bool = False              # LLM重排
    llm_reranking_sample_size: int = 30      # 重排候选数
    top_n_retrieval: int = 10                # 最终返回数
    parallel_requests: int = 1               # 并发请求数
    api_provider: str = "dashscope"           # API供应商
    answering_model: str = "qwen-turbo-latest"
    full_context: bool = False # 全量上下文
```

### 5.2 预定义配置

`process-questions --config` 实际可选项（定义于 [src/pipeline.py](src/pipeline.py#L220-L235)）：

| 配置名 | 说明 | LLM / 关键参数 |
|--------|------|---------------|
| `base` | 基础配置 | GPT-4o-mini，`parallel_requests=10` |
| `pdr` | 父文档检索 | GPT-4o |
| `max` | LLM 重排 + 并行 | Qwen-Turbo，`parallel_requests=4` |
| `minimax` | LLM 重排 + 自定义路径 | MiniMax-M2.7，`parallel_requests=4` |

`process-reports --config` 用的是另一组配置 `preprocess_configs`（[src/pipeline.py](src/pipeline.py#L188-L189)），只控制离线预处理：

| 配置名 | 说明 |
|--------|------|
| `ser_tab` | 解析结果写入 `03_reports_markdown_ser_tab/`（包含表格序列化产物） |
| `no_ser_tab` | 解析结果写入 `03_reports_markdown/`（不含表格序列化） |

---

## 6. 目录结构

```
QueryDocs/
├── main.py                    # CLI 入口（6 个子命令）
├── requirements.txt
├── env                        # 环境变量模板（需改名为 .env）
├── setup.py
├── questions.json             # 待回答问题
├── subset.csv                 # 公司清单 (sha1, file_name, company_name)
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # 主管道 (Pipeline / RunConfig / PipelineConfig / configs)
│   ├── pdf_mineru.py          # PDF 解析 (MinerU API)
│   ├── parsed_reports_merging.py # 解析结果规整
│   ├── text_splitter.py       # 类型感知分块
│   ├── process_chunks.py      # 批量分块处理
│   ├── ingestion.py           # FAISS + BM25 索引构建
│   ├── retrieval.py           # VectorRetriever / BM25Retriever / HybridRetriever
│   ├── reranking.py           # LLMReranker / JinaReranker
│   ├── questions_processing.py # 问答主逻辑 (QuestionsProcessor)
│   ├── tables_serialization.py # 表格 LLM 序列化 (TableSerializer)
│   ├── prompts.py             # 提示词 + Pydantic Schema
│   ├── api_requests.py        # 多 API 处理器 (BaseOpenai / BaseDashscope / BaseMiniMax / BaseIBM / BaseGemini)
│   ├── api_request_parallel_processor.py # 并发限流批处理
│   │
│   └── api/                   # FastAPI Web 服务
│       ├── __init__.py
│       ├── main.py
│       ├── models.py          # QARequest / QAResponse / Reference
│       ├── pipeline_wrapper.py # 惰性 Pipeline 单例
│       └── routers/
│           ├── __init__.py
│           └── qa.py          # POST /api/qa/ask
│
├── data/
│   └── stock_data/
│       ├── subset.csv
│       ├── questions.json
│       ├── pdf_reports/       # 原始 PDF
│       ├── debug_data/
│       │   ├── 03_reports_markdown/         # MinerU 解析结果
│       │   ├── 03_reports_markdown_ser_tab/ # 表格序列化版本
│       │   ├── chunked_reports/             # 分块结果
│       │   └── databases/
│       │       ├── vector_dbs/              # FAISS 向量库
│       │       └── bm25_dbs/                # BM25 pickle
│       └── answers_*.json     # 批量问答结果
│
├── frontend/                  # React + TypeScript + Vite 前端
│   ├── src/
│   ├── package.json
│   └── vite.config.ts
│
└── docs/
    ├── architecture.md        # 本文档
    └── src_modules_overview.md # src 模块速查
```

---

## 7. 依赖技术栈

| 类别 | 库/工具 | 用途 |
|------|---------|------|
| **PDF解析** | `mineru` (API) | 云端PDF解析，支持表格/公式/OCR |
| **向量数据库** | `faiss-cpu` | 高效向量相似度检索 |
| **BM25** | `rank-bm25` | 传统关键词检索 |
| **嵌入** | `dashscope`, `openai` | 文本向量化 |
| **LLM** | `openai`, `dashscope`, `anthropic`, `google-generativeai` | 多API支持 |
| **分块** | `tiktoken` | Token级分块 |
| **结构化** | `pydantic` | 输出验证 |
| **并发** | `concurrent.futures`, `asyncio` | 并行请求 |
| **Web** | `fastapi`, `uvicorn` | API服务 |
| **编码** | `tiktoken` | Token计数 |

---

## 8. 关键设计决策

### 8.1智能分块策略

**问题**: 简单按chunk_size分块会切断表格、标题等重要结构。

**解决方案**: 按内容类型分别处理：
- **原子类型** (`title`, `table`, `image`): 不可切断，保证结构完整
- **可分割类型** (`paragraph`, `list`): 按自然句子边界分割
- 标题与后续内容组合，超长时以标题为前缀
- **表格超长分割时保留表头**（`<thead>`），确保列语义不丢失

### 8.2 混合检索 + LLM重排

**问题**: 向量检索对专有名词、数字不敏感；关键词检索无法理解语义。

**解决方案**:
```
向量检索 (候选) → LLM相关性评分 → 加权融合分数
```

### 8.3 父文档检索 (Parent Document Retrieval)

**问题**: 分块导致上下文碎片化，答案跨越多个chunk。

**解决方案**:
- 检索时返回**完整页面**而非分块
- 解决需要完整段落理解的问题

### 8.4 表格序列化

**问题**: 表格HTML无法被向量检索正确命中。

**解决方案**:
- 使用LLM将表格HTML转为结构化信息块
- 包含核心实体、表头、完整描述
- 使表格内容可被检索

### 8.5 指标严格匹配

**问题**: LLM可能返回"相关但不等价"的指标，导致幻觉。

**解决方案** (`AnswerWithRAGContextNumberPrompt`):
- 严格校验指标定义是否完全一致
- 拒绝范围不匹配、概念不同的指标
- 无法确定时默认返回 `N/A`

### 8.6 页码引用校验

**问题**: LLM可能幻觉出不存在的页码引用。

**解决方案**:
- `_validate_page_references()` 校验引用的页码是否真实存在
- 自动过滤幻觉页码
- 补充检索结果中的top页

### 8.7 多公司比较问题处理

**问题**: 比较问题涉及多个公司，单次检索无法处理。

**解决方案**:
- `RephrasedQuestionsPrompt` 将比较问题分解为单公司问题
- 并行处理各公司子问题
- `ComparativeAnswerPrompt` 汇总生成最终比较结论

---

## 9. API接口

### 9.1 REST API

**入口文件**: `src/api/main.py`（FastAPI app）

| Method | Path | 说明 |
|--------|------|------|
| GET | `/` | 服务信息 |
| GET | `/api/health` | 健康检查，返回 `{"status": "ok"}` |
| POST | `/api/qa/ask` | 单问题推理（**注意是 `/ask` 而非 `/answer`**） |

**请求** `POST /api/qa/ask`：

```json
{
  "question": "请简要总结某公司2022年主营业务内容。",
  "kind": "string"
}
```

`kind` 可选值：`string`（默认）/ `number` / `boolean` / `names`。

**响应** `QAResponse`（定义于 [src/api/models.py](src/api/models.py)）：

```json
{
  "step_by_step_analysis": "分步推理（≥5 步 ≥150 字）",
  "reasoning_summary": "推理摘要",
  "relevant_pages": [1, 2, 3],
  "final_answer": "最终答案或 'N/A'",
  "references": [
    { "pdf_sha1": "stock_10001", "page_index": 1 }
  ]
}
```

> `references` 字段是后端从 `QAResponse` 中**额外补的**结构化引用（`pdf_sha1` 取自 `subset.csv`，`page_index` 对齐 `relevant_pages`），便于前端做页码跳转。`relevant_pages` 仅给页码列表，`references` 才把"页码"映射回"具体 PDF"。

**curl 示例**：

```bash
curl -X POST http://localhost:8000/api/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "请简要总结某公司2022年主营业务内容。", "kind": "string"}'
```

### 9.2 CLI 批量处理

```bash
# 1) 解析 PDF
python main.py parse-pdfs --parallel --max-workers 10

# 2) （可选）表格序列化
python main.py serialize-tables --max-workers 10

# 3) 报告分块
cd data/stock_data
python ../../main.py process-reports --config ser_tab

# 4) 构建向量库 / BM25
python main.py build-vectors
python main.py build-bm25

# 5) 批量问答（回到项目根目录）
cd ../..
python main.py process-questions --config minimax
```

---

## 10. 局限性

1. **API依赖**: 高度依赖外部API服务可用性
2. **GPU需求**: PDF解析在GPU环境下效率更高
3. **无测试代码**: 缺少单元测试和集成测试
4. **`Pipeline` 类与 CLI 子命令错位**: 当前 `Pipeline` 只承担在线推理编排，离线索引（解析 / 分块 / 向量化 / BM25）由 CLI 直接驱动对应模块，原 `__main__` 块里调用的 `chunk_reports / create_vector_dbs` 在 `Pipeline` 类中并不存在

---

*文档版本: 2.1*
*最后更新: 2026-06-01*