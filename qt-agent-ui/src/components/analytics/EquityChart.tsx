import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface EquityPoint {
  timestamp: string;
  equity: number;
  balance: number;
}

interface EquityChartProps {
  data: EquityPoint[];
}

export default function EquityChart({ data }: EquityChartProps) {
  if (!data || data.length === 0) {
    return (
      <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6">
        <h3 className="text-lg font-semibold mb-4">Equity Curve</h3>
        <div className="flex items-center justify-center h-64 text-slate-500 dark:text-slate-400">
          No equity data available
        </div>
      </div>
    );
  }

  const currentEquity = data[data.length - 1]?.equity || 0;
  const startEquity = data[0]?.equity || 0;
  const change = currentEquity - startEquity;
  const changePercent = startEquity !== 0 ? (change / startEquity) * 100 : 0;

  // Format data for Recharts
  const chartData = data.map(point => ({
    timestamp: new Date(point.timestamp).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
    equity: point.equity
  }));

  return (
    <div className="rounded-2xl border border-slate-200 dark:border-slate-800 bg-white dark:bg-slate-900 p-6">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="text-lg font-semibold">Equity Curve</h3>
          <div className="flex items-baseline gap-3 mt-2">
            <span className="text-3xl font-bold">${currentEquity.toFixed(2)}</span>
            <span className={`text-sm font-medium ${changePercent >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
              {changePercent >= 0 ? '+' : ''}{changePercent.toFixed(2)}%
            </span>
          </div>
        </div>
      </div>

      {/* Chart */}
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData}>
            <CartesianGrid strokeDasharray="3 3" stroke="currentColor" className="text-slate-200 dark:text-slate-800" />
            <XAxis 
              dataKey="timestamp" 
              stroke="currentColor" 
              className="text-slate-500 dark:text-slate-400"
              tick={{ fontSize: 12 }}
            />
            <YAxis 
              stroke="currentColor" 
              className="text-slate-500 dark:text-slate-400"
              tick={{ fontSize: 12 }}
              tickFormatter={(value) => `$${value.toFixed(0)}`}
            />
            <Tooltip 
              contentStyle={{ 
                backgroundColor: 'rgb(30 41 59)',
                border: '1px solid rgb(51 65 85)',
                borderRadius: '8px',
                color: 'white'
              }}
              formatter={(value: number) => [`$${value.toFixed(2)}`, 'Equity']}
            />
            <Line 
              type="monotone" 
              dataKey="equity" 
              stroke="rgb(16 185 129)" 
              strokeWidth={2}
              dot={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-3 gap-4 mt-6 pt-6 border-t border-slate-200 dark:border-slate-800">
        <div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Start</div>
          <div className="font-semibold">${startEquity.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Current</div>
          <div className="font-semibold">${currentEquity.toFixed(2)}</div>
        </div>
        <div>
          <div className="text-xs text-slate-500 dark:text-slate-400 mb-1">Change</div>
          <div className={`font-semibold ${change >= 0 ? 'text-emerald-500' : 'text-red-500'}`}>
            {change >= 0 ? '+' : ''}${change.toFixed(2)}
          </div>
        </div>
      </div>
    </div>
  );
}
