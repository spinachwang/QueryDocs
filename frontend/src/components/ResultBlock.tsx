import React from 'react';

interface ResultBlockProps {
  title: string;
  badge?: string;
  icon: React.ReactNode;
  variant: 'reasoning' | 'summary' | 'pages' | 'answer';
  children: React.ReactNode;
}

export const ResultBlock: React.FC<ResultBlockProps> = ({ title, badge, icon, variant, children }) => {
  return (
    <div className={`result-block b-${variant}`}>
      <div className="block-header">
        <div className="block-icon">{icon}</div>
        <span className="block-title">{title}</span>
        {badge && <span className="block-badge">{badge}</span>}
      </div>
      <div className="block-body">{children}</div>
    </div>
  );
};