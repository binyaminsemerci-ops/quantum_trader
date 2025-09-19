import React from 'react';

type TradeSummary = {
  id: string | number;
  symbol?: string;
  side?: string;
};

function TradeList({ trades }: { trades: TradeSummary[] }) {
  return (
    <div style={{ marginTop: '2rem' }}>
      <h2>Trades</h2>
      <ul>
        {trades.map((t) => (
          <li key={String(t.id)}>
            <a href={`/trades/${t.id}`}>{t.symbol ?? '-'} — {t.side ?? '-'}</a>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default TradeList;
