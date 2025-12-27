// components/EquityChart.tsx
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface EquityChartProps {
  data: any[];
  height?: number;
}

export default function EquityChart({ data, height = 400 }: EquityChartProps) {
  const CustomTooltip = ({ active, payload }: any) => {
    if (active && payload && payload.length) {
      return (
        <div className="bg-quantum-card border border-quantum-border rounded-lg p-3 shadow-lg">
          <p className="text-quantum-text text-sm">
            Equity: <span className="font-bold text-quantum-accent">${payload[0].value.toFixed(2)}</span>
          </p>
          {payload[0].payload.timestamp && (
            <p className="text-quantum-muted text-xs mt-1">
              {new Date(payload[0].payload.timestamp).toLocaleString()}
            </p>
          )}
        </div>
      );
    }
    return null;
  };

  return (
    <div className="bg-quantum-card border border-quantum-border rounded-lg p-6">
      <h3 className="text-lg font-semibold text-quantum-text mb-4">Equity Curve</h3>
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#2a2a38" />
          <XAxis 
            dataKey="timestamp" 
            hide 
            stroke="#9ca3af"
          />
          <YAxis 
            stroke="#9ca3af"
            tick={{ fill: '#9ca3af', fontSize: 12 }}
          />
          <Tooltip content={<CustomTooltip />} />
          <Line 
            type="monotone" 
            dataKey="equity" 
            stroke="#22c55e" 
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 6, fill: '#22c55e' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
