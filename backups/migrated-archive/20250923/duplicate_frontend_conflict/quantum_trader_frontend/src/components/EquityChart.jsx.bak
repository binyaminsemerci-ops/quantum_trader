// frontend/src/components/EquityChart.jsx
import { useState } from "react";
import { useDashboardData } from "../hooks/useDashboardData.jsx";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  Brush,
  ReferenceArea,
} from "recharts";

function ChartSkeleton() {
  return (
    <div className="p-4 animate-pulse">
      <div className="h-6 bg-gray-200 rounded w-40 mb-4"></div>
      <div className="h-48 bg-gray-200 rounded"></div>
    </div>
  );
}

export default function EquityChart() {
  const { data } = useDashboardData();
  const chart = data.chart || [];

  // Zoom states
  const [zoom, setZoom] = useState({ startIndex: 0, endIndex: null });
  const [refArea, setRefArea] = useState(null);

  if (!chart.length) return <ChartSkeleton />;

  // Reset zoom
  const resetZoom = () => {
    setZoom({ startIndex: 0, endIndex: null });
  };

  return (
    <div className="p-4">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold">📈 Equity Curve</h2>
        <button
          onClick={resetZoom}
          className="px-3 py-1 bg-gray-200 dark:bg-gray-700 rounded"
        >
          Reset Zoom
        </button>
      </div>

      <ResponsiveContainer width="100%" height={300}>
        <LineChart
          data={chart}
          onMouseDown={(e) => {
            if (e) setRefArea({ x1: e.activeLabel, x2: null });
          }}
          onMouseMove={(e) => {
            if (refArea && e) {
              setRefArea((prev) => ({ ...prev, x2: e.activeLabel }));
            }
          }}
          onMouseUp={() => {
            if (refArea && refArea.x1 && refArea.x2 && refArea.x1 !== refArea.x2) {
              const start = chart.findIndex((c) => c.date === refArea.x1);
              const end = chart.findIndex((c) => c.date === refArea.x2);
              if (start !== -1 && end !== -1) {
                setZoom({
                  startIndex: Math.min(start, end),
                  endIndex: Math.max(start, end),
                });
              }
            }
            setRefArea(null);
          }}
        >
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis dataKey="date" />
          <YAxis />
          <Tooltip />
          <Line
            type="monotone"
            dataKey="equity"
            stroke="#2563eb"
            strokeWidth={2}
            dot={false}
          />

          {/* Brush for quick zoom */}
          <Brush dataKey="date" height={20} stroke="#2563eb" />

          {/* Reference area for selection */}
          {refArea && refArea.x1 && refArea.x2 ? (
            <ReferenceArea
              x1={refArea.x1}
              x2={refArea.x2}
              strokeOpacity={0.3}
            />
          ) : null}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
