import { useEffect, useState } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

interface Trade {
  id: string;
  timestamp: string;
  symbol: string;
  side: string;
  pnl: number;
}

export default function PnLChart() {
  const [data, setData] = useState<{ day: string; pnl: number }[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchTrades = async () => {
      try {
        const response = await fetch("http://localhost:8000/api/trades");
        
        if (!response.ok) {
          console.warn("Trades endpoint returned error:", response.status);
          setLoading(false);
          return;
        }
        
        const trades: Trade[] = await response.json();

        // Group by day and sum PnL
        const pnlByDay: { [key: string]: number } = {};
        
        trades.forEach((trade) => {
          const date = new Date(trade.timestamp);
          const day = date.toLocaleDateString("no-NO", { 
            month: "short", 
            day: "numeric" 
          });
          
          if (!pnlByDay[day]) {
            pnlByDay[day] = 0;
          }
          pnlByDay[day] += trade.pnl || 0;
        });

        // Convert to array and sort by date
        const chartData = Object.entries(pnlByDay)
          .map(([day, pnl]) => ({ day, pnl }))
          .slice(-14); // Last 14 days

        setData(chartData);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching trades:", error);
        setLoading(false);
      }
    };

    fetchTrades();
    const interval = setInterval(fetchTrades, 60000); // Update every minute

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="h-full grid place-items-center text-slate-400">
        Loading PnL data...
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="h-full flex flex-col items-center justify-center text-slate-400">
        <div className="text-2xl font-bold text-slate-500">$0.00</div>
        <div className="text-xs mt-1">No trade data available</div>
        <div className="text-xs mt-2 text-slate-500">Waiting for completed trades...</div>
      </div>
    );
  }

  const totalPnL = data.reduce((sum, item) => sum + item.pnl, 0);

  return (
    <div className="h-full flex flex-col">
      <div className="mb-2">
        <div className={`text-2xl font-bold ${totalPnL >= 0 ? "text-emerald-600" : "text-rose-600"}`}>
          ${totalPnL.toFixed(2)}
        </div>
        <div className="text-xs text-slate-500">Total P&L (Last {data.length} days)</div>
      </div>
      
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" opacity={0.3} />
          <XAxis 
            dataKey="day" 
            stroke="var(--text-secondary)"
            tick={{ fontSize: 10 }}
          />
          <YAxis 
            stroke="var(--text-secondary)"
            tick={{ fontSize: 10 }}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'var(--panel)', 
              border: '1px solid var(--border)',
              borderRadius: '4px',
              fontSize: '12px'
            }}
            formatter={(value: number) => [`$${value.toFixed(2)}`, 'PnL']}
          />
          <Bar dataKey="pnl" radius={[4, 4, 0, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.pnl >= 0 ? "#10b981" : "#ef4444"} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
