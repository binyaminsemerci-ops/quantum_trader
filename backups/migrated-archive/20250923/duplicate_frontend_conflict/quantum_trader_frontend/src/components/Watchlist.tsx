<<<<<<< Updated upstream
import { useDashboardData } from '../hooks/useDashboardData';

export default function Watchlist(): JSX.Element {
  const { data } = useDashboardData();
  const symbols = data?.stats?.pnl_per_symbol || {};

=======
<<<<<<<< Updated upstream:frontend/src/components/Watchlist.jsx
// Auto-generated re-export stub
export { default } from './Watchlist.tsx';
========
// Demo: Mock feed fra WebSocket data.stats.active_symbols
import { useDashboardData } from "../hooks/useDashboardData.tsx";

export default function Watchlist() {
  const { data } = useDashboardData();
  const symbols = data.stats?.pnl_per_symbol || {};
>>>>>>> Stashed changes
  return (
    <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg mt-4">
      <h3 className="font-bold mb-2">Watchlist</h3>
      <table className="min-w-full">
        <thead>
          <tr>
            <th className="text-left">Symbol</th>
            <th className="text-left">PnL</th>
          </tr>
        </thead>
        <tbody>
          {Object.entries(symbols).map(([sym, pnl]) => (
            <tr key={sym}>
              <td>{sym}</td>
<<<<<<< Updated upstream
              <td className={Number(pnl) >= 0 ? 'text-green-600' : 'text-red-600'}>{String(pnl)}</td>
=======
              <td className={pnl >= 0 ? "text-green-600" : "text-red-600"}>{pnl}</td>
>>>>>>> Stashed changes
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
<<<<<<< Updated upstream
=======
>>>>>>>> Stashed changes:frontend/src/components/Watchlist.tsx
>>>>>>> Stashed changes
