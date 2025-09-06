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
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchEquity() {
      try {
        const res = await fetch("http://127.0.0.1:8000/chart");
        const data = await res.json();
        setEquity(data);
      } catch (err) {
        console.error("Feil ved henting av equity:", err);
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
            tickFormatter={(ts) =>
              new Date(ts).toLocaleDateString("no-NO", {
                day: "2-digit",
                month: "2-digit",
              })
            }
          />
          <YAxis domain={["auto", "auto"]} />
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
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
