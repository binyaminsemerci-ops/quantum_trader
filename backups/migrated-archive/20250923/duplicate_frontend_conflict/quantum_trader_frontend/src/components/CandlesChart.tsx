<<<<<<< Updated upstream
import { useEffect, useState } from 'react';

type Props = { symbol?: string; limit?: number };

export default function CandlesChart({ symbol = 'BTCUSDT', limit = 50 }: Props): JSX.Element {
  const [loading, setLoading] = useState(true);
  const [candles, setCandles] = useState<any[]>([]);

  useEffect(() => {
    let canceled = false;
    async function fetchCandles() {
      try {
        const res = await fetch(`http://127.0.0.1:8000/api/candles?symbol=${symbol}&limit=${limit}`);
        const data = await res.json();
        if (!canceled) setCandles(data.candles || []);
      } catch (err) {
        console.error('Failed to fetch candles', err);
      } finally {
        if (!canceled) setLoading(false);
      }
    }
    fetchCandles();
    return () => { canceled = true; };
  }, [symbol, limit]);

  if (loading) return <div className="p-4 bg-white rounded shadow animate-pulse">Loading candles…</div>;
  if (!candles.length) return <div className="p-4 bg-white rounded shadow">No candles</div>;

  return (
    <div className="p-4 bg-white rounded shadow">
      <h2 className="text-xl font-bold mb-2">📊 {symbol} Candles</h2>
      <div className="text-sm text-gray-600">Showing last {candles.length} candles.</div>
    </div>
  );
}
=======
<<<<<<<< Updated upstream:frontend/src/components/CandlesChart.jsx
// Auto-generated re-export stub
export { default } from './CandlesChart.tsx';
========
import { useEffect, useState } from "react";
import {
  ComposedChart,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Bar,
  Line,
} from "recharts";

export default function CandlesChart({ symbol = "BTCUSDT", limit = 50 }) {
  const [candles, setCandles] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchCandles() {
      try {
        const res = await fetch(
          `http://127.0.0.1:8000/api/candles?symbol=${symbol}&limit=${limit}`
        );
  const data = await res.json();
        setCandles(data.candles || []);
      } catch (err) {
        console.error("Failed to fetch candles", err);
      } finally {
        setLoading(false);
      }
    }
    fetchCandles();
  }, [symbol, limit]);

  if (loading) {
    return (
      <div className="p-4 bg-white dark:bg-gray-800 rounded shadow animate-pulse">
        <div className="h-6 bg-gray-300 dark:bg-gray-700 rounded w-40 mb-4"></div>
        <div className="h-48 bg-gray-200 dark:bg-gray-700 rounded"></div>
      </div>
    );
  }

  if (!candles.length) {
    return (
      <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
        No candles found for {symbol}.
      </div>
    );
  }

  // Recharts candlestick hack: use Bar for body, Line for high/low wick
  const chartData = candles.map((c) => ({
    ...c,
    color: c.close > c.open ? "#22c55e" : "#ef4444", // green if bullish, red if bearish
    body: Math.abs(c.close - c.open),
    wickLow: Math.min(c.open, c.close),
    wickHigh: Math.max(c.open, c.close),
  }));

  return (
    <div className="p-4 bg-white dark:bg-gray-800 rounded shadow">
      <h2 className="text-xl font-bold mb-4">📊 {symbol} Candlestick Chart</h2>
      <ResponsiveContainer width="100%" height={400}>
        <ComposedChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="timestamp" hide />
          <YAxis domain={["auto", "auto"]} />
          <Tooltip />

          {/* Wicks */}
          <Line
            type="monotone"
            dataKey="wickLow"
            stroke="#888"
            dot={false}
            isAnimationActive={false}
          />
          <Line
            type="monotone"
            dataKey="wickHigh"
            stroke="#888"
            dot={false}
            isAnimationActive={false}
          />

          {/* Candle body */}
          <Bar
            dataKey="body"
            barSize={6}
            shape={(props) => {
              const { x, y, width, height, payload } = props;
              const open = payload.open;
              const close = payload.close;
              const color = payload.color;
              const top = Math.min(open, close);
              const bottom = Math.max(open, close);
              const candleHeight = Math.max(1, Math.abs(y - (y + height)));

              return (
                <rect
                  x={x}
                  y={y}
                  width={width}
                  height={candleHeight}
                  fill={color}
                  stroke={color}
                />
              );
            }}
          />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
>>>>>>>> Stashed changes:frontend/src/components/CandlesChart.tsx
>>>>>>> Stashed changes
