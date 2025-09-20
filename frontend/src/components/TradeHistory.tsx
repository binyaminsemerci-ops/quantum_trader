// @ts-nocheck
import React from 'react';

export default function TradeHistory() {
  return (
    <div className="trade-history-placeholder">
      <h4>Trade History Placeholder</h4>
      <p>Temporary placeholder while migrating to TypeScript.</p>
    </div>
  );
}
// @ts-nocheck
import React from 'react';

export default function TradeHistory({ trades = [] }) {
  return (
    <div className="trade-history">
      <h4>Trade History</h4>
      <ul>
        {(trades || []).map((t, i) => (
          <li key={t?.id ?? t?._id ?? i}>{t?.symbol ?? 'N/A'} - {t?.qty ?? 0}</li>
        ))}
      </ul>
    </div>
  );
}
import React, { useState, useEffect } from 'react';
import '../styles/TradeHistory.css';

type Trade = {
  trade_id: string | number;
  symbol: string;
  side: string;
  quantity: number;
  price?: number | null;
  timestamp: string;
};

const TradeHistory: React.FC = () => {
  const [trades, setTrades] = useState<Trade[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        setLoading(true);
        const response = await fetch('http://localhost:8000/trades');
        if (!response.ok) throw new Error(`Error: ${response.status}`);
        const data = await response.json();
        setTrades(data ?? []);
        setLoading(false);
      } catch (err: any) {
        setError(err.message ?? 'Unknown error');
        setLoading(false);
      }
    };

    fetchTrades();
    const timer = setInterval(fetchTrades, 10000);
    return () => clearInterval(timer);
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
            {trades.map((trade) => (
              <tr key={trade.trade_id}>
                <td>{trade.trade_id}</td>
                <td>{trade.symbol}</td>
                <td className={trade.side === 'BUY' ? 'buy' : 'sell'}>{trade.side}</td>
                <td>{trade.quantity}</td>
                <td>{trade.price ?? 'Market'}</td>
                <td>{new Date(trade.timestamp).toLocaleString()}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
};

export default TradeHistory;
