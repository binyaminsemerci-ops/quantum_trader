<<<<<<< Updated upstream
export default function ChartView({ data }: { data?: any[] }): JSX.Element {
  if (!data || data.length === 0) {
    return (
      <div className="p-6 bg-white shadow rounded-lg flex items-center justify-center h-64">
        <p className="text-gray-500">No chart data available</p>
      </div>
    );
  }

  return (
    <div className="p-6 bg-white shadow rounded-lg">
      <h2 className="text-lg font-semibold text-gray-700 mb-4">Performance Chart</h2>
      <div className="h-64 flex items-center justify-center text-gray-400">[ Chart placeholder ]</div>
    </div>
  );
}
=======
// Auto-generated re-export stub
export { default } from './ChartView.tsx';
>>>>>>> Stashed changes
