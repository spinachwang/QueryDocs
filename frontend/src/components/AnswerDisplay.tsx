import React from 'react';

interface AnswerDisplayProps {
  answer: string;
}

export const AnswerDisplay: React.FC<AnswerDisplayProps> = ({ answer }) => {
  return (
    <div className="answer-text">
      {answer.split('\n').map((line, i) => (
        <p key={i} style={{ marginBottom: line ? '12px' : 0 }}>{line || ' '}</p>
      ))}
      <div className="answer-disclaimer">
        ⚠️ 本答案由 AI 综合多篇研报生成，仅供参考，不构成投资建议。投资有风险，决策需审慎。
      </div>
    </div>
  );
};