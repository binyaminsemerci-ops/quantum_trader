import PriceChartRecharts from './PriceChartRecharts';
import type { OHLCV } from '../types';
import useCandlesWS from '../hooks/useCandlesWS';
import useCandlesPoll from '../hooks/useCandlesPoll';

type Props = { symbol?: string; limit?: number };

export default function PriceChart({ symbol = 'BTCUSDC', limit = 100 }: Props) {
  // Small wrapper that currently re-uses existing CandlesChart component.
  // Later we can swap in Recharts / TradingView here for a richer candlestick view.
  // Prefer websocket live feed; fall back to polling
  const ws = useCandlesWS('/ws/dashboard');
  const poll = useCandlesPoll(symbol, limit, 5000);

  const data: OHLCV[] = ws.data && ws.data.length ? ws.data : poll.data;

  if (!data || !data.length) return <div className="p-4 bg-white rounded shadow">No candle data</div>;

  return (
    <div className="p-4 bg-white rounded shadow">
  <h2 className="text-xl font-bold mb-2">📊 {symbol} Candles</h2>
      <PriceChartRecharts data={data} />
    </div>
  );
}
