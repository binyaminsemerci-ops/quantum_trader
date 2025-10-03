import React, { useState, useEffect } from 'react';
import axios from 'axios';

function formatTimestamp(ts?: string) {
  if (!ts) return '-';
  try {
    const d = new Date(ts);
    return d.toLocaleString();
  } catch {
    return ts;
  }
}

type Signal = {
  id?: string | number;
  symbol?: string;
  signal?: string | null;
  confidence?: number | null;
  timestamp?: string;
  executed?: boolean;
};

const SignalsList: React.FC = () => {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [executingSignal, setExecutingSignal] = useState<string | null>(null);

  useEffect(() => { fetchSignals(); }, []);

  async function fetchSignals(): Promise<void> {
    setLoading(true);
    try {
      const res = await axios.get<Signal[]>('/api/v1/signals/recent?limit=20');
      setSignals(res.data ?? []);
    } catch (err: unknown) {
      console.error('Error fetching signals', err);
      setError('Failed to load trading signals');
    } finally {
      setLoading(false);
    }
  }

  async function executeSignal(signal: Signal) {
    if (!signal.symbol || !signal.signal) return;
    setExecutingSignal(signal.symbol);
    try {
      await axios.post('/api/v1/trading/manual-trade', {
        symbol: signal.symbol,
        action: signal.signal.toUpperCase(),
        force: false
      });
      await fetchSignals();
    } catch (err: unknown) {
      console.error('Error executing signal', err);
      alert('Failed to execute signal');
    } finally {
      setExecutingSignal(null);
    }
  }

  async function generateSignal(symbol: string) {
    setExecutingSignal(symbol);
    try {
      await axios.post(`/api/v1/trading/analyze/${symbol}`);
      await fetchSignals();
    } catch (err: unknown) {
      console.error('Error generating signal', err);
      alert('Failed to generate signal');
    } finally {
      setExecutingSignal(null);
    }
  }

  if (loading) return <div className="signals-list loading">Loading signals...</div>;
  if (error) return <div className="signals-list error">{error}</div>;

  return (
    <div className="signals-list">
      <h3>AI Trading Signals</h3>

      <div className="generate-signals mb-4">
        <h4>Generate New Signal</h4>
        <div className="flex gap-2">
          <button onClick={() => generateSignal('BTCUSDT')} disabled={executingSignal === 'BTCUSDT'}>BTC Signal</button>
          <button onClick={() => generateSignal('ETHUSDT')} disabled={executingSignal === 'ETHUSDT'}>ETH Signal</button>
          <button onClick={() => generateSignal('XRPUSDT')} disabled={executingSignal === 'XRPUSDT'}>XRP Signal</button>
        </div>
      </div>

      {signals.length === 0 ? (
        <p>No signals found. Generate your first signal!</p>
      ) : (
        <table className="w-full table-auto border-collapse">
          <thead>
            <tr>
              <th>Symbol</th>
              <th>Signal</th>
              <th>Confidence</th>
              <th>Timestamp</th>
              <th>Executed</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {signals.map((s) => (
              <tr key={String(s.id)}>
                <td>{s.symbol}</td>
                <td>{s.signal}</td>
                <td>{typeof s.confidence === 'number' ? `${Math.round(s.confidence * 100)}%` : '—'}</td>
                <td>{formatTimestamp(s.timestamp)}</td>
                <td>{s.executed ? 'Yes' : 'No'}</td>
                <td>
                  {!s.executed && (
                    <button onClick={() => executeSignal(s)} disabled={executingSignal === s.symbol}>Execute</button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <div className="mt-2">
        <button onClick={fetchSignals}>Refresh</button>
      </div>
    </div>
  );
};

export default SignalsList;
