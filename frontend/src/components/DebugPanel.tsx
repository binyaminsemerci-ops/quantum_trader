import React, { useEffect, useState } from 'react';
import { checkHealth, fetchTrades } from '../lib/api';
import type { Trade as SharedTrade } from '../types';

type HealthResponse = { ok: boolean; data?: unknown; error?: unknown };

export default function DebugPanel(): JSX.Element {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [trades, setTrades] = useState<SharedTrade[] | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [err, setErr] = useState<string | null>(null);

  const refresh = async () => {
    setLoading(true);
    setErr(null);
    try {
      const h = (await checkHealth()) as HealthResponse;
      setHealth(h);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e);
      setErr(`health error: ${msg}`);
    }
    try {
      const t = await fetchTrades();
      setTrades(Array.isArray(t) ? (t as SharedTrade[]) : null);
    } catch (e: unknown) {
      // keep the original console message for debugging
      console.error('fetchTrades debug error', e);
      const msg = e instanceof Error ? e.message : String(e);
      setErr((prev) => (prev ? `${prev}; trades error: ${msg}` : `trades error: ${msg}`));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div style={{ padding: 12, border: '1px solid #ddd', margin: 12, fontFamily: 'sans-serif' }}>
      <h3>Debug: backend /api</h3>
      <button onClick={refresh} disabled={loading}>
        Refresh
      </button>
      <div style={{ marginTop: 8 }}>
        <strong>Health:</strong>
        <pre>{JSON.stringify(health, null, 2)}</pre>
      </div>
      <div style={{ marginTop: 8 }}>
        <strong>Trades (count):</strong> {Array.isArray(trades) ? trades.length : String(trades)}
        <pre style={{ maxHeight: 300, overflow: 'auto' }}>{JSON.stringify(trades, null, 2)}</pre>
      </div>
      {err && (
        <div style={{ color: 'red', marginTop: 8 }}>
          <strong>Error:</strong> {err}
        </div>
      )}
    </div>
  );
}