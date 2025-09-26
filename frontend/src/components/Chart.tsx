import { useEffect, useState } from 'react';
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts';
import api from '../utils/api';

type DataPoint = { timestamp: string | number; equity: number };

export default function Chart(): JSX.Element {
  const [equity, setEquity] = useState<DataPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchEquity() {
      try {
        const resp = await api.getChart();
        if (resp && 'data' in resp && Array.isArray(resp.data)) {
          const raw = resp.data as unknown[];
          const points: DataPoint[] = raw.map((item, idx) => {
            if (typeof item === 'number') return { timestamp: idx, equity: item };
            if (Array.isArray(item)) {
              // [ts, open, high, low, close, volume] style
              const maybeClose = Number(item[4]);
              return { timestamp: item[0] ?? idx, equity: Number.isFinite(maybeClose) ? maybeClose : 0 };
            }
            // object form
            const o = item as Record<string, unknown>;
            const equityVal = Number(o.equity ?? o.close ?? o.value ?? NaN);
            const timestamp = o.timestamp ?? o.date ?? idx;
            return { timestamp, equity: Number.isFinite(equityVal) ? equityVal : 0 };
          });
          setEquity(points);
        } else {
          setEquity([]);
        }
      } catch (err) {
        // eslint-disable-next-line no-console
        console.error('Error fetching equity:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchEquity();
  }, []);

  if (loading) return <p className="text-gray-500">⏳ Laster equity-kurve...</p>;
  if (!equity || equity.length === 0) return <p className="text-gray-500">⚠️ Ingen equity-data funnet</p>;

  return (
    <div className="bg-white p-6 shadow rounded-2xl">
      <h3 className="text-xl font-bold mb-4">📈 Equity-kurve</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={equity}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={(ts: string | number) =>
              new Date(Number(ts)).toLocaleDateString('no-NO', {
                day: '2-digit',
                month: '2-digit',
              })
            }
          />
          <YAxis domain={["auto", "auto"]} />
          <Tooltip labelFormatter={(ts: string | number) => new Date(Number(ts)).toLocaleString('no-NO')} />
          <Line type="monotone" dataKey="equity" stroke="#2563eb" strokeWidth={2} dot={false} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
