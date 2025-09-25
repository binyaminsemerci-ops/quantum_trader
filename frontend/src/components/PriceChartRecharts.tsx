import { useMemo } from 'react';
import { ResponsiveContainer, ComposedChart, XAxis, YAxis, Tooltip, CartesianGrid, Bar, Line } from 'recharts';
import type { OHLCV } from '../types';

type Props = { data: OHLCV[]; height?: number };

// Convert OHLCV union into a normalized candle shape
function normalize(data: OHLCV[]) {
  return data
    .map((d) => {
      if (typeof d === 'number') return null;
      // timestamp may be stored in `timestamp` or `date` depending on source
      const t = (d as any).timestamp ?? (d as any).date ?? undefined;
      return {
        time: t ?? undefined,
        open: d.open ?? 0,
        high: d.high ?? 0,
        low: d.low ?? 0,
        close: d.close ?? 0,
        volume: d.volume ?? 0,
      };
    })
    .filter(Boolean) as Array<{ time?: string | number; open: number; high: number; low: number; close: number; volume: number }>;
}

export default function PriceChartRecharts({ data, height = 320 }: Props): JSX.Element {
  const candles = useMemo(() => normalize(data), [data]);

  return (
    <div style={{ width: '100%', height }}>
      <ResponsiveContainer width="100%" height="100%">
        <ComposedChart data={candles}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="time" tickFormatter={(v) => String(v).toString()} />
          <YAxis domain={["dataMin", "dataMax"]} />
          <Tooltip />
          {/* Use Bar for volume and Line for close price */}
          <Bar dataKey="volume" barSize={20} fill="#8884d8" yAxisId="right" />
          <Line type="monotone" dataKey="close" stroke="#ff7300" dot={false} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  );
}
