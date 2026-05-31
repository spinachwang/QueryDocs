# RAG-CY 架构文档

## 1. 项目概述

**项目名称**: RAG-CY - 年报智能问答系统
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
│  ┌─────────────────┐     ┌─────────────────────┐  ┌──────────────────────┐ │
│  │   main.py       │     │   app_streamlit.py │  │   src/api/main.py    │ │
│  │   (CLI入口)     │     │   (Web界面)          │  │   (FastAPI服务)      │ │
│ └────────┬────────┘     └──────────┬──────────┘  └──────────┬───────────┘ │
└───────────┼─────────────────────────┼────────────────────────┼─────────────┘
            │                         │                        │
            ▼                         ▼                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                             管道编排层 (Pipeline)                             │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │ src/pipeline.py ││
│  │  • parse_pdf_reports()      - 并行PDF解析                                ││
│  │  • chunk_reports()          - 智能文本分块                               ││
│  │  • create_vector_dbs()      - 构建FAISS向量库                           ││
│  │  • create_bm25_db()         - 构建BM25索引                              ││
│  │  • process_questions()      - 批量问题处理                               ││
│  │  • answer_single_question() - 单问题推理 ││
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
| `main.py` | CLI命令行入口，支持 `download-models`, `parse-pdfs`, `serialize-tables`, `process-reports`, `process-questions` 等命令 |
| `src/api/main.py` | FastAPI Web服务，提供RESTful API接口 |
| `app_streamlit.py` | (已废弃) Web界面入口 |

### 3.2 管道编排 (Pipeline)

**文件**: `src/pipeline.py`

核心类 `Pipeline` 协调整个RAG管道流程：

```
PDF输入 → 解析 → 分块 → 向量化 → 存储 → 检索 → 重排 → LLM推理 → 答案输出
```

**关键配置**:
- `RunConfig`: 运行时参数配置（是否使用序列化表格、父文档检索、LLM重排、并发数等）
- `PipelineConfig`: 路径配置（数据目录、输出目录、数据库路径）

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
| `table` | 按HTML标签分割(`<tr>`, `<td>`)，保持表格结构 |
| `list` | 作为整体保留，或按句子边界分割 |
| `image` | 保留图片路径引用 |

#### 关键算法

```python
def split_content_list_v2(content_list_v2_path, output_path,
                           chunk_size=300, overlap=50, pdf_name=None):
    # 1. 按页遍历MinerU的content_list_v2.json
    # 2. 遇到标题暂存，与后续内容组合
    # 3. 超长内容按句子分割，保持语义完整
    # 4. 表格按HTML标签边界分割
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

| 配置名 | 说明 |
|--------|------|
| `base` | 基础配置，GPT-4o-mini |
| `pdr` | 启用父文档检索 |
| `max` | 最佳性能配置: 父文档检索 + LLM重排 + Qwen Turbo |
| `max_nst_o3m` | o3-mini模型版本 |
| `gemini_thinking` | Gemini全上下文模式 |
| `minimax` | MiniMax-M2.7模型配置 |

---

## 6. 目录结构

```
RAG-cy/
├── main.py                    # CLI入口
├── requirements.txt           # 依赖
├── setup.py
│
├── src/
│   ├── __init__.py
│   ├── pipeline.py            # 主管道
│   ├── pdf_mineru.py          # PDF解析 (MinerU API)
│   ├── text_splitter.py       # 文本分块 (类型感知)
│   ├── ingestion.py           # 索引构建 (FAISS/BM25)
│   ├── retrieval.py           # 检索器
│   ├── reranking.py           # LLM重排
│   ├── questions_processing.py # 问题处理
│   ├── api_requests.py        # 多API处理器
│   ├── prompts.py            # 提示词模板
│   ├── tables_serialization.py # 表格序列化
│   ├── process_chunks.py      # 批量分块处理
│   │
│   └── api/ # FastAPI Web服务
│       ├── __init__.py
│       ├── main.py
│       ├── models.py
│       ├── pipeline_wrapper.py
│       └── routers/
│           ├── __init__.py
│           └── qa.py
│
├── data/                      # 数据目录
│   └── stock_data/
│       ├── subset.csv         # 公司列表
│       ├── questions.json    # 问题列表
│       ├── pdf_reports/       # 原始PDF
│       └── debug_data/       # 中间结果
│           └── *_reports_markdown/
│           └── chunked_reports/
│           └── databases/
│
└── docs/
    └── architecture.md       # 本文档
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

###9.1 REST API

**文件**: `src/api/main.py`

```
GET  /api/health          - 健康检查
GET  /                    - 服务信息
POST /api/qa/answer        - 单问题推理
```

**请求示例**:
```bash
curl -X POST http://localhost:8000/api/qa/answer \
  -H "Content-Type: application/json" \
  -d '{"question": "请简要总结某公司2022年主营业务内容。", "kind": "string"}'
```

### 9.2 CLI批量处理

```bash
# 解析PDF
python main.py parse-pdfs --parallel --max-workers 10

# 处理报告（分块+向量化）
cd data/stock_data
python ../../main.py process-reports --config ser_tab

# 处理问题
python ../../main.py process-questions --config max_nst_o3m
```

---

## 10. 局限性

1. **API依赖**: 高度依赖外部API服务可用性
2. **GPU需求**: PDF解析在GPU环境下效率更高
3. **无测试代码**: 缺少单元测试和集成测试

---

*文档版本: 2.0*
*最后更新: 2026-05-31*