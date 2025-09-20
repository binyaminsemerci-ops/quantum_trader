// @ts-nocheck
import React, { useState, useEffect } from 'react';
import '../styles/TradeHistory.css';

function TradeHistory() {
  const [trades, setTrades] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let mounted = true;
    const fetchTrades = async () => {
      try {
        if (mounted) setLoading(true);
        const response = await fetch('http://localhost:8000/trades');

        if (!response.ok) {
          throw new Error(`Error: ${response.status}`);
        }

        const data = await response.json();
        if (mounted && Array.isArray(data)) setTrades(data);
        if (mounted) setLoading(false);
      } catch (err) {
        if (mounted) {
          setError(err?.message || String(err));
          setLoading(false);
        }
      }
    };

    fetchTrades();

    const timer = setInterval(fetchTrades, 10000);

    return () => {
      mounted = false;
      clearInterval(timer);
    };
  }, []);

  if (loading && trades.length === 0) return <div className="loading">Loading trades...</div>;
  if (error) return <div className="error">Error loading trades: {error}</div>;

  return (
    <div className="trade-history">
      <h2>Trade History</h2>
      {trades.length === 0 ? (
        <p>No trades found</p>
      ) : (
        <table>
          <thead>
            <tr>
              <th>ID</th>
              <th>Symbol</th>
              <th>Side</th>
              <th>Quantity</th>
              <th>Price</th>
              <th>Timestamp</th>
            </tr>
          </thead>
          <tbody>
            {trades.map((trade, idx) => {
              const id = trade?.trade_id ?? trade?.id ?? idx;
              const ts = trade?.timestamp ? new Date(trade.timestamp).toLocaleString() : 'N/A';
              return (
                <tr key={id}>
                  <td>{id}</td>
                  <td>{trade?.symbol ?? '—'}</td>
                  <td className={(trade?.side === 'BUY') ? 'buy' : 'sell'}>
                    {trade?.side ?? '—'}
                  </td>
                  <td>{trade?.quantity ?? '—'}</td>
                  <td>{trade?.price ?? 'Market'}</td>
                  <td>{ts}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default TradeHistory;
