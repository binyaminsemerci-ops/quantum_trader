<<<<<<< Updated upstream
import { useDashboardData } from '../hooks/useDashboardData';

export default function RiskCards(): JSX.Element | null {
  const { data } = useDashboardData();
  const risk = data?.stats?.risk;
=======
<<<<<<<< Updated upstream:frontend/src/components/RiskCards.jsx
// Auto-generated re-export stub
export { default } from './RiskCards.tsx';
========
import { useDashboardData } from "../hooks/useDashboardData";

export default function RiskCards() {
  const { data } = useDashboardData();
  const risk = data.stats?.risk;
>>>>>>> Stashed changes
  if (!risk) return null;

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
      <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
        <h3>Max Trade Exposure</h3>
        <p className="text-xl font-bold">{risk.max_trade_exposure}</p>
      </div>
      <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
        <h3>Daily Loss Limit</h3>
        <p className="text-xl font-bold text-red-600">{risk.daily_loss_limit}</p>
      </div>
      <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg">
        <h3>Exposure per Symbol</h3>
        {Object.entries(risk.exposure_per_symbol || {}).map(([sym, val]) => (
          <div key={sym} className="flex justify-between text-sm">
            <span>{sym}</span>
<<<<<<< Updated upstream
            <span>{String(val)}</span>
=======
            <span>{val}</span>
>>>>>>> Stashed changes
          </div>
        ))}
      </div>
    </div>
  );
}
<<<<<<< Updated upstream
=======
>>>>>>>> Stashed changes:frontend/src/components/RiskCards.tsx
>>>>>>> Stashed changes
