// using automatic JSX runtime
import TradeTable from '../components/TradeTable';
import { useDashboardData } from '../hooks/useDashboardData';

export default function Trades(): JSX.Element {
  const { data } = useDashboardData();
  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">📑 Handler</h2>
      <TradeTable trades={data.trades} />
    </div>
  );
}
