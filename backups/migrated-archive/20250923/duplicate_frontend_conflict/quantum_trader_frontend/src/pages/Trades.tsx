<<<<<<< Updated upstream
// using automatic JSX runtime
import TradeTable from '../components/TradeTable';
import { useDashboardData } from '../hooks/useDashboardData';

type Trade = { id?: string | number; symbol?: string; side?: string; qty?: number; price?: number; status?: string; timestamp?: string };

export default function Trades(): JSX.Element {
  const { data } = useDashboardData();
  return (
    <div className="p-6 space-y-6">
      <h2 className="text-2xl font-bold">📑 Handler</h2>
      <TradeTable trades={data?.trades as Trade[] | undefined} />
    </div>
  );
}
=======
// Auto-generated re-export stub
export { default } from './Trades.tsx';
>>>>>>> Stashed changes
