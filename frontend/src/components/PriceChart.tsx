import React from 'react';
import CandlesChart from './CandlesChart';

type Props = { symbol?: string; limit?: number };

export default function PriceChart({ symbol = 'BTCUSDT', limit = 100 }: Props) {
  // Small wrapper that currently re-uses existing CandlesChart component.
  // Later we can swap in Recharts / TradingView here for a richer candlestick view.
  return (
    <div>
      <CandlesChart symbol={symbol} limit={limit} />
    </div>
  );
}
