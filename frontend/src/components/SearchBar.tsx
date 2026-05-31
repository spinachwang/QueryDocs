import React from 'react';

interface SearchBarProps {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  loading: boolean;
}

export const SearchBar: React.FC<SearchBarProps> = ({ value, onChange, onSubmit, loading }) => {
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      onSubmit();
    }
  };

  return (
    <div className="search-wrap">
      <input
        className="search-input"
        type="text"
        placeholder="例如：中芯国际在晶圆制造行业中的地位如何？"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={loading}
      />
      <svg className="search-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <circle cx="11" cy="11" r="8"></circle><path d="m21 21-4.35-4.35"></path>
      </svg>
      <button className="search-btn" onClick={onSubmit} disabled={loading}>
        {loading ? '查询中...' : '查询答案'}
      </button>
    </div>
  );
};