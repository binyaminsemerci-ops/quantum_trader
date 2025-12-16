import { useEffect, useState } from "react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from "recharts";

interface Candle {
  timestamp: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface PriceChartProps {
  symbol?: string;
  limit?: number;
}

export default function PriceChart({ symbol = "BTCUSDT", limit = 100 }: PriceChartProps) {
  const [data, setData] = useState<Candle[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchCandles = async () => {
      try {
        const response = await fetch(`http://localhost:8000/candles?symbol=${symbol}&limit=${limit}`);
        if (!response.ok) {
          throw new Error(`HTTP error! status: ${response.status}`);
        }
        const result = await response.json();
        // Backend returns {symbol: string, candles: array}
        const candles = result?.candles || [];
        setData(candles);
      } catch (error) {
        console.error("Error fetching candles:", error);
        setData([]);
      } finally {
        setLoading(false);
      }
    };

    fetchCandles();
    const interval = setInterval(fetchCandles, 30000); // Update every 30s

    return () => clearInterval(interval);
  }, [symbol, limit]);

  if (loading) {
    return (
      <div className="h-full grid place-items-center text-slate-400">
        Loading chart...
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="h-full grid place-items-center text-slate-400">
        No price data available
      </div>
    );
  }

  const chartData = data.map((candle) => {
    // Handle both formats: timestamp (number) or time (ISO string)
    const timeValue = candle.timestamp 
      ? new Date(candle.timestamp).toLocaleTimeString("no-NO", { hour: "2-digit", minute: "2-digit" })
      : candle.time 
        ? new Date(candle.time).toLocaleTimeString("no-NO", { hour: "2-digit", minute: "2-digit" })
        : "N/A";
    
    return {
      time: timeValue,
      price: candle.close,
    };
  });

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={chartData} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
        <XAxis 
          dataKey="time" 
          stroke="var(--text-secondary)"
          tick={{ fontSize: 11 }}
          interval="preserveStartEnd"
        />
        <YAxis 
          stroke="var(--text-secondary)"
          tick={{ fontSize: 11 }}
          domain={['dataMin - 10', 'dataMax + 10']}
        />
        <Tooltip 
          contentStyle={{ 
            backgroundColor: 'var(--panel)', 
            border: '1px solid var(--border)',
            borderRadius: '4px',
            fontSize: '12px'
          }}
          labelStyle={{ color: 'var(--text)' }}
        />
        <Line 
          type="monotone" 
          dataKey="price" 
          stroke="#10b981" 
          strokeWidth={2}
          dot={false}
          activeDot={{ r: 4 }}
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
