import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  ChartOptions
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface TrendChartProps {
  data: number[];
  labels: string[];
  title?: string;
  color?: string;
}

export default function TrendChart({ 
  data, 
  labels, 
  title, 
  color = '#10b981' 
}: TrendChartProps) {
  const chartData = {
    labels,
    datasets: [
      {
        label: title || 'Trend',
        data,
        borderColor: color,
        backgroundColor: `${color}20`,
        tension: 0.4,
        fill: true,
        pointRadius: 0,
        pointHoverRadius: 6,
        borderWidth: 2
      }
    ]
  };

  const options: ChartOptions<'line'> = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        display: false
      },
      tooltip: {
        backgroundColor: '#1f2937',
        titleColor: '#fff',
        bodyColor: '#fff',
        borderColor: '#374151',
        borderWidth: 1
      }
    },
    scales: {
      x: {
        grid: {
          color: '#374151'
        },
        ticks: {
          color: '#9ca3af'
        }
      },
      y: {
        grid: {
          color: '#374151'
        },
        ticks: {
          color: '#9ca3af'
        }
      }
    }
  };

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
      {title && (
        <h3 className="text-lg font-semibold mb-4 text-gray-200">{title}</h3>
      )}
      <div className="h-64">
        <Line data={chartData} options={options} />
      </div>
    </div>
  );
}
