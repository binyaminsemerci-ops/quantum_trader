import React, { useEffect, useState } from "react";
import { api } from "../utils/api";
import { LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";
import type { OHLCV } from "../types";

const OHLCVChart: React.FC = () => {
  const [data, setData] = useState<{ time: string; close: number }[]>([]);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const res = await api.getChart();
        const src: OHLCV[] = (res?.data && Array.isArray(res.data)) ? res.data as OHLCV[] : [];
        if (!mounted) return;
        const parsed = src.map((d) => ({
    time: new Date(Number(d.timestamp ?? Date.now())).toLocaleTimeString(),
          close: Number(d.close ?? 0),
        }));
        setData(parsed);
      } catch (e) {
        console.error('Failed to load OHLCV chart', e);
        if (mounted) setData([]);
      }
    })();
    return () => { mounted = false; };
  }, []);

  if (!data || data.length === 0) return <div>No chart data</div>;

  return (
    <div>
      <h2>BTCUSDT Price (last 20 mins)</h2>
      <LineChart width={600} height={300} data={data}>
        <XAxis dataKey="time" />
        <YAxis />
        <Tooltip />
        <CartesianGrid stroke="#ccc" />
        <Line type="monotone" dataKey="close" stroke="#8884d8" />
      </LineChart>
    </div>
  );
};

export default OHLCVChart;
