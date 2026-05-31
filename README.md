# RAG-CY - 年报智能问答系统

> 基于检索增强生成（RAG）技术的公司年报智能问答系统。

[English](#english) | [中文](#中文)

---

## English

### Project Overview

RAG-CY is a RAG-based question answering system designed to answer questions about company annual reports. It combines advanced PDF parsing, intelligent chunking, hybrid retrieval, and large language model technologies to provide accurate, structured answers from annual report documents.

### RAG Technologies Used

#### 1. PDF Parsing (MinerU API)
- Cloud-based PDF parsing with MinerU API
- Supports text, tables, formulas, and OCR
- Outputs structured JSON (`content_list_v2.json`) preserving document structure

#### 2. Intelligent Chunking Strategy
- **Type-aware chunking**: Different strategies for different content types
  - Atomic types (`title`, `table`, `image`): Never split, preserve structure integrity
  - Splittable types (`paragraph`, `list`): Split at sentence boundaries
- **Title-content combination**: Titles are combined with subsequent content as prefixes
- **Table-aware splitting**: Tables split at HTML tag boundaries (`<tr>`, `<td>`)
- Configurable chunk size (default 300 tokens) and overlap (default 50 tokens)

#### 3. Hybrid Retrieval
- **Vector retrieval** (FAISS + DashScope/OpenAI embeddings): Captures semantic similarity
- **BM25 keyword retrieval**: Exact keyword matching for numbers and proper nouns
- **HybridRetriever**: Combines both with weighted scoring

#### 4. LLM-based Reranking
- **LLMReranker**: Uses LLM (Qwen/GPT-4o-mini/MiniMax) to score document relevance
- **Weighted fusion**: `combined_score = llm_weight * relevance_score + vector_weight * distance`
- Reduces hallucinations by filtering irrelevant documents

#### 5. Parent Document Retrieval
- Retrieves complete pages instead of small chunks
- Solves context fragmentation issues
- Ideal for questions requiring full paragraph understanding

#### 6. Table Serialization
- Converts table HTML into structured information blocks via LLM
- Includes: core entity, relevant headers, full description
- Makes table content searchable via embeddings

#### 7. Structured Output with Chain-of-Thought
- Pydantic models validate output format
- Step-by-step reasoning (CoT) in `step_by_step_analysis` field
- Different prompt templates for different answer types:
  - `NumberPrompt`: Strict metric matching, rejects non-equivalent indicators
  - `BooleanPrompt`: True/False answers
  - `NamesPrompt`: Entity lists (names, positions, products)
  - `StringPrompt`: Free-text summaries
  - `ComparativePrompt`: Multi-company comparisons

#### 8. Query Routing for Comparative Questions
- Detects multiple companies in questions
- Decomposes comparative questions into single-company sub-questions
- Parallel processing with final comparative conclusion

### Optimization Highlights

#### Performance Optimizations
| Optimization | Implementation |
|--------------|----------------|
| **Parallel processing** | ThreadPoolExecutor for batch question processing |
| **QPS protection** | `max_workers=1` for DashScope to avoid rate limits |
| **Batch embedding** | 25 documents per batch for vector DB ingestion |
| **Incremental saving** | Checkpoint saving during batch processing |

#### Retrieval Optimizations
| Optimization | Implementation |
|--------------|----------------|
| **Company-based filtering** | Pre-filter by company name before retrieval |
| **Hybrid scoring** | Combines vector + keyword + LLM scores |
| **Relevance calibration** | LLM reranking for better context selection |
| **Page reference validation** | Filters hallucinated page numbers |

#### Answer Quality Optimizations
| Optimization | Implementation |
|--------------|----------------|
| **Strict metric matching** | NumberPrompt rejects non-equivalent indicators |
| **Hallucination detection** | Validates page references against retrieval results |
| **Citation verification** | Ensures cited pages actually exist in context |
| **Incremental improvement** | Multi-round CoT reasoning |

### Technical Challenges & Solutions

#### Challenge 1: Table Structure Preservation
**Problem**: Simple chunking by token count would split tables, breaking their structure.

**Solution**: Type-aware chunking with table-aware splitting:
```python
ATOMIC_TYPES = {'title', 'table', 'image', 'page_header', 'page_number'}
SPLITTABLE_TYPES = {'paragraph', 'list'}

# Tables split at HTML tag boundaries
html_parts = re.split(r'(<tr>|</tr>|<td>|</td>)', part)
```

#### Challenge 2: Context Fragmentation
**Problem**: Small chunks lose paragraph-level context needed for comprehensive answers.

**Solution**: Parent Document Retrieval returns complete pages:
```python
# Instead of returning small chunks, return full pages
if return_parent_pages:
    result = {"page": parent_page["page"], "text": parent_page["text"]}
```

#### Challenge 3: Metric Hallucination
**Problem**: LLM might return "related but not equivalent" metrics (e.g., revenue vs. net profit).

**Solution**: Strict indicator matching in NumberPrompt:
```
**Strict indicator matching requirement:**
1. Only accept if context indicator meaning is EXACTLY equivalent
2. Reject if: scope mismatch, proxy indicators, requires calculation
3. Default to 'N/A' if any doubt about equivalence
```

#### Challenge 4: Page Reference Hallucination
**Problem**: LLM might cite non-existent page numbers.

**Solution**: Page reference validation:
```python
def _validate_page_references(self, claimed_pages, retrieval_results):
    retrieved_pages = [r['page'] for r in retrieval_results]
    validated = [p for p in claimed_pages if p in retrieved_pages]
    # Filter hallucinated pages, supplement from top results
```

#### Challenge 5: Comparative Question Routing
**Problem**: Comparative questions involve multiple companies, cannot be processed in single retrieval.

**Solution**: Query decomposition pipeline:
```python
# 1. Detect companies in question
# 2. Decompose into single-company sub-questions
# 3. Process in parallel
# 4. Generate comparative conclusion
```

### Tech Stack

| Category | Technology | Purpose |
|----------|------------|---------|
| PDF Parsing | MinerU API | Cloud-based PDF parsing |
| Vector DB | FAISS | Similarity search |
| Keyword Retrieval | rank-bm25 | BM25 indexing |
| Embeddings | DashScope / OpenAI | Text vectorization |
| LLM | Qwen / GPT-4o / MiniMax | Answer generation |
| API | FastAPI | RESTful API service |
| Concurrency | concurrent.futures | Parallel processing |
| Validation | Pydantic | Output structure validation |

### Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure API keys in .env
cp .env.example .env

# Parse PDFs
python main.py parse-pdfs --parallel --max-workers 10

# Process reports (chunking + vectorization)
cd data/stock_data
python ../../main.py process-reports --config ser_tab

# Process questions
python ../../main.py process-questions --config max

# Or start API server
cd src/api
uvicorn main:app --reload --port 8000
```

### Project Structure

```
RAG-cy/
├── main.py                 # CLI entry point
├── src/
│   ├── pipeline.py        # Pipeline orchestration
│   ├── pdf_mineru.py      # PDF parsing (MinerU API)
│   ├── text_splitter.py   # Type-aware chunking
│   ├── ingestion.py       # Vector/BM25 indexing
│   ├── retrieval.py       # Retrieval (Vector/BM25/Hybrid)
│   ├── reranking.py       # LLM reranking
│   ├── questions_processing.py  # Question processing
│   ├── api_requests.py    # Multi-API processor
│   ├── prompts.py         # Prompt templates
│   ├── tables_serialization.py  # Table serialization
│   ├── process_chunks.py  # Batch chunk processing
│   └── api/              # FastAPI service
└── data/stock_data/      # Data directory
```

---

## 中文

### 项目介绍

RAG-CY 是一个基于 RAG（检索增强生成）技术的年报智能问答系统。项目通过结合高级PDF解析、智能文本分块、混合检索和大语言模型技术，实现对公司年报的智能问答，提供准确、结构化的答案。

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
- **表格感知分割**：按HTML标签边界（`<tr>`, `<td>`）分割表格
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
- 思维链推理：`step_by_step_analysis`字段进行分步推理
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

###难点及克服方法

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

```bash
# 安装依赖
pip install -r requirements.txt

# 配置API密钥
cp .env.example .env

# 解析PDF
python main.py parse-pdfs --parallel --max-workers 10

# 处理报告（分块+向量化）
cd data/stock_data
python ../../main.py process-reports --config ser_tab

# 处理问题
python ../../main.py process-questions --config max

# 或启动API服务
cd src/api
uvicorn main:app --reload --port 8000
```

### 项目结构

```
RAG-cy/
├── main.py                 # CLI入口
├── src/
│   ├── pipeline.py        # 管道编排
│   ├── pdf_mineru.py      # PDF解析 (MinerU API)
│   ├── text_splitter.py   # 类型感知分块
│   ├── ingestion.py       # 向量/BM25索引构建
│   ├── retrieval.py       # 检索器 (向量/BM25/混合)
│   ├── reranking.py       # LLM重排
│   ├── questions_processing.py  # 问题处理
│   ├── api_requests.py    # 多API处理器
│   ├── prompts.py         # 提示词模板
│   ├── tables_serialization.py  # 表格序列化
│   ├── process_chunks.py  # 批量分块处理
│   └── api/              # FastAPI服务
└── data/stock_data/      # 数据目录
```

### 许可证

MIT