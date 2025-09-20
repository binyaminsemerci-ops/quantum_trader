import React, { useEffect, useState } from "react";
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

type Candle = { open: number; close: number; timestamp?: string | number };

export default function CandlesChart({
  symbol = "BTCUSDT",
  limit = 50,
}: {
  symbol?: string;
  limit?: number;
}): JSX.Element {
  const [candles, setCandles] = useState<Candle[]>([]);
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

  const chartData = candles.map((c) => ({
    ...c as any,
    color: (c.close ?? 0) > (c.open ?? 0) ? "#22c55e" : "#ef4444",
    body: Math.abs((c.close ?? 0) - (c.open ?? 0)),
    wickLow: Math.min(c.open ?? 0, c.close ?? 0),
    wickHigh: Math.max(c.open ?? 0, c.close ?? 0),
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
            shape={(props: any) => {
              const { x, y, width, height } = props;
              const payload = props.payload ?? {};
              const open = payload.open ?? 0;
              const close = payload.close ?? 0;
              const color = payload.color ?? "#888";
              const top = Math.min(open, close);
              const bottom = Math.max(open, close);
              const candleHeight = Math.max(1, height ?? 1);

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
