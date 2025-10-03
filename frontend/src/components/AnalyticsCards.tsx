// using automatic JSX runtime
import { useDashboardData } from '../hooks/useDashboardData';

export default function AnalyticsCards(): JSX.Element | null {
  const { data } = useDashboardData();
  const analytics = data?.stats?.analytics;
  if (!analytics) return null;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
      <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
        <h3>Win Rate</h3>
        <p className="text-xl font-bold">{analytics.win_rate}%</p>
      </div>
      <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
        <h3>Sharpe Ratio</h3>
        <p className="text-xl font-bold">{analytics.sharpe_ratio}</p>
      </div>
      <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
        <h3>Total Trades</h3>
        <p className="text-xl font-bold">{analytics.trades_count}</p>
      </div>
    </div>
  );
}
