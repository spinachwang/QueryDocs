import { useState, useCallback } from 'react';
import type { QAResponse } from '../types';

const API_BASE = 'http://localhost:8000';

export function useQA() {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QAResponse | null>(null);

  const ask = useCallback(async (question: string) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`${API_BASE}/api/qa/ask`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question, kind: 'string' }),
      });

      if (!res.ok) {
        throw new Error(`API error: ${res.status}`);
      }

      const data: QAResponse = await res.json();
      setResult(data);
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, []);

  const clear = useCallback(() => {
    setResult(null);
    setError(null);
  }, []);

  return { loading, error, result, ask, clear };
}