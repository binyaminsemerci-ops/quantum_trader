<<<<<<< Updated upstream
import { useEffect, useState } from 'react';
import * as Recharts from 'recharts';

// Recharts typings sometimes conflict with the project's JSX settings.
// Cast components to `any` for a conservative migration (keeps runtime behavior,
// avoids complex typing churn in a small PR). Replace with proper types later.
const ResponsiveContainer: any = Recharts.ResponsiveContainer as any;
const LineChart: any = Recharts.LineChart as any;
const Line: any = Recharts.Line as any;
const XAxis: any = Recharts.XAxis as any;
const YAxis: any = Recharts.YAxis as any;
const CartesianGrid: any = Recharts.CartesianGrid as any;
const Tooltip: any = Recharts.Tooltip as any;

type EquityPoint = { timestamp: number | string; equity: number };

export default function Chart(): JSX.Element {
  const [equity, setEquity] = useState<EquityPoint[]>([]);
=======
import React, { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";

export default function Chart() {
  const [equity, setEquity] = useState([]);
>>>>>>> Stashed changes
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchEquity() {
      try {
<<<<<<< Updated upstream
        const res = await fetch('http://127.0.0.1:8000/chart');
        const data = await res.json();
        setEquity(data as EquityPoint[]);
      } catch (err) {
        console.error('Error fetching equity:', err);
=======
        const res = await fetch("http://127.0.0.1:8000/chart");
        const data = await res.tson();
        setEquity(data);
      } catch (err) {
        console.error("Feil ved henting av equity:", err);
>>>>>>> Stashed changes
      } finally {
        setLoading(false);
      }
    }
    fetchEquity();
  }, []);

  if (loading) {
    return <p className="text-gray-500">⏳ Laster equity-kurve...</p>;
  }

  if (!equity || equity.length === 0) {
    return <p className="text-gray-500">⚠️ Ingen equity-data funnet</p>;
  }

  return (
    <div className="bg-white p-6 shadow rounded-2xl">
      <h3 className="text-xl font-bold mb-4">📈 Equity-kurve</h3>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={equity}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
<<<<<<< Updated upstream
            tickFormatter={(ts: number | string) =>
              new Date(Number(ts)).toLocaleDateString('no-NO', {
                day: '2-digit',
                month: '2-digit',
=======
            tickFormatter={(ts) =>
              new Date(ts).toLocaleDateString("no-NO", {
                day: "2-digit",
                month: "2-digit",
>>>>>>> Stashed changes
              })
            }
          />
          <YAxis domain={["auto", "auto"]} />
<<<<<<< Updated upstream
          <Tooltip labelFormatter={(ts: number | string) => new Date(Number(ts)).toLocaleString('no-NO')} />
          <Line type="monotone" dataKey="equity" stroke="#2563eb" strokeWidth={2} dot={false} />
=======
          <Tooltip
            labelFormatter={(ts) => new Date(ts).toLocaleString("no-NO")}
          />
          <Line
            type="monotone"
            dataKey="equity"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
          />
>>>>>>> Stashed changes
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
