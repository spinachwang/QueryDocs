import React from 'react';

interface PageCardProps {
  source: string;
  title: string;
  page: number;
  score?: number;
}

export const PageCard: React.FC<PageCardProps> = ({ source, title, page, score }) => {
  return (
    <div className="page-card">
      <div className="page-source">{source}</div>
      <div className="page-title">{title}</div>
      <div className="page-meta">
        <span className="page-date">Page {page}</span>
        {score && <span className="page-score">相关度 {score}%</span>}
      </div>
    </div>
  );
};