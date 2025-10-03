import { memo } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

interface ChartData {
  timestamp: string;
  price: number;
  volume?: number;
}

interface SimpleChartProps {
  data: ChartData[];
  title: string;
  color?: string;
  height?: number;
}

function SimpleChartComponent({ 
  data, 
  title, 
  color = '#3b82f6',
  height = 200 
}: SimpleChartProps) {
  
  if (!data || data.length === 0) {
    return (
      <div className={`bg-gray-800 rounded-lg p-6 h-[${height}px]`}>
        <h3 className="text-lg font-semibold text-gray-300 mb-4">{title}</h3>
        <div className="flex items-center justify-center h-32">
          <p className="text-gray-500">Ingen chart data tilgjengelig</p>
        </div>
      </div>
    );
  }

  return (
    <div className={`bg-gray-800 rounded-lg p-6 h-[${height}px]`}>
      <h3 className="text-lg font-semibold text-gray-300 mb-4">{title}</h3>
      <ResponsiveContainer width="100%" height={height - 100}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
          <XAxis 
            dataKey="timestamp" 
            stroke="#9ca3af"
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => new Date(value).toLocaleTimeString()}
          />
          <YAxis 
            stroke="#9ca3af"
            tick={{ fontSize: 12 }}
            tickFormatter={(value) => `$${value.toLocaleString()}`}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: '#1f2937', 
              border: '1px solid #374151',
              borderRadius: '6px'
            }}
            labelFormatter={(value) => new Date(value).toLocaleString()}
            formatter={(value: any) => [`$${value.toLocaleString()}`, 'Price']}
          />
          <Line 
            type="monotone" 
            dataKey="price" 
            stroke={color}
            strokeWidth={2}
            dot={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

const SimpleChart = memo(SimpleChartComponent, (prev, next) => {
  // Unngå re-render hvis lengde og siste punkt er likt.
  if (prev.data === next.data) return true;
  if (prev.data.length !== next.data.length) return false;
  const lastPrev = prev.data[prev.data.length - 1];
  const lastNext = next.data[next.data.length - 1];
  return lastPrev?.timestamp === lastNext?.timestamp && lastPrev?.price === lastNext?.price;
});

export default SimpleChart;