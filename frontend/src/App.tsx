import { useState } from 'react'
import './index.css'
import { SearchBar } from './components/SearchBar'
import { ResultBlock } from './components/ResultBlock'
import { PageCard } from './components/PageCard'
import { AnswerDisplay } from './components/AnswerDisplay'
import { useQA } from './hooks/useQA'
import type { QAResponse } from './types'

const StepIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="13 17 18 12 13 7"></polyline><polyline points="6 17 11 12 6 7"></polyline>
  </svg>
)

const SummaryIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="8" y1="6" x2="21" y2="6"></line><line x1="8" y1="12" x2="21" y2="12"></line><line x1="8" y1="18" x2="21" y2="18"></line>
    <line x1="3" y1="6" x2="3.01" y2="6"></line><line x1="3" y1="12" x2="3.01" y2="12"></line><line x1="3" y1="18" x2="3.01" y2="18"></line>
  </svg>
)

const PagesIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
    <polyline points="14 2 14 8 20 8"></polyline>
  </svg>
)

const AnswerIcon = () => (
  <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="10"></circle>
    <line x1="12" y1="8" x2="12" y2="12"></line>
    <line x1="12" y1="16" x2="12.01" y2="16"></line>
  </svg>
)

function parseReasoningSteps(text: string): string[] {
  const lines = text.split('\n').filter(l => l.trim());
  const steps: string[] = [];
  let current = '';

  for (const line of lines) {
    if (/^\d+[．、.、]/.test(line.trim()) || line.trim().startsWith('1.') || line.trim().startsWith('1）')) {
      if (current) steps.push(current);
      current = line;
    } else if (/^[２３４５６７８９]/.test(line.trim())) {
      if (current) steps.push(current);
      current = line;
    } else {
      current += ' ' + line;
    }
  }
  if (current) steps.push(current);

  if (steps.length === 0 && text) {
    return text.split('\n').filter(l => l.trim());
  }

  return steps;
}

function App() {
  const [query, setQuery] = useState('')
  const { loading, error, result, ask } = useQA()

  const handleSubmit = () => {
    if (query.trim()) {
      ask(query)
    }
  }

  const renderResults = (data: QAResponse) => {
    const steps = parseReasoningSteps(data.step_by_step_analysis);

    return (
      <>
        <ResultBlock
          title="分布式推理"
          badge={`${steps.length} 步`}
          icon={<StepIcon />}
          variant="reasoning"
        >
          <div className="reasoning-steps">
            {steps.map((step, i) => (
              <div key={i} className="reasoning-step">
                <div className="step-num">{i + 1}</div>
                <div className="step-text">{step}</div>
              </div>
            ))}
          </div>
        </ResultBlock>

        <ResultBlock
          title="推理摘要"
          badge="摘要"
          icon={<SummaryIcon />}
          variant="summary"
        >
          <p className="summary-lead">{data.reasoning_summary}</p>
        </ResultBlock>

        <ResultBlock
          title="相关页面"
          badge={`${data.references.length} 篇`}
          icon={<PagesIcon />}
          variant="pages"
        >
          <div className="pages-grid">
            {data.references.map((ref, i) => (
              <PageCard
                key={i}
                source={ref.pdf_sha1}
                title={`Page ${ref.page_index}`}
                page={ref.page_index}
              />
            ))}
          </div>
        </ResultBlock>

        <ResultBlock
          title="最终答案"
          badge="综合"
          icon={<AnswerIcon />}
          variant="answer"
        >
          <AnswerDisplay answer={data.final_answer} />
        </ResultBlock>
      </>
    );
  };

  return (
    <>
      <header>
        <div className="header-inner">
          <span className="wordmark">RAG<span>QA</span></span>
          <span className="header-badge">SecDB</span>
        </div>
      </header>

      <main>
        <p className="hero-label">证券研究知识库</p>
        <h1 className="hero-title">基于研报的智能问答系统</h1>
        <p className="hero-sub">输入您关心的问题，自动检索相关研报，生成结构化的推理过程与投资建议。</p>

        <SearchBar
          value={query}
          onChange={setQuery}
          onSubmit={handleSubmit}
          loading={loading}
        />

        {error && (
          <div className="query-echo" style={{ borderColor: 'red', color: 'red' }}>
            错误：{error}
          </div>
        )}

        <div className="results">
          {result && (
            <>
              <div className="query-echo">
                当前查询：<strong>{query}</strong>
              </div>
              {renderResults(result)}
            </>
          )}
        </div>

        {loading && (
          <div className="skeleton">
            <div className="skeleton-row"><div className="skeleton-line short"></div><div className="skeleton-line"></div><div className="skeleton-line"></div></div>
            <div className="skeleton-row"><div className="skeleton-line short"></div><div className="skeleton-line"></div></div>
            <div className="skeleton-row"><div className="skeleton-line short"></div><div className="skeleton-line"></div></div>
            <div className="skeleton-row"><div className="skeleton-line short"></div><div className="skeleton-line"></div></div>
          </div>
        )}
      </main>
    </>
  )
}

export default App