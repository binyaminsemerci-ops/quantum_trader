<<<<<<< Updated upstream
import { useEffect, useState } from 'react';
import { api } from '../utils/api';

type SpotBalance = { free?: number; asset?: string } | any;
type FuturesBalance = { balance?: number; asset?: string } | any;

export default function BalanceCard(): JSX.Element {
  const [spot, setSpot] = useState<SpotBalance | null>(null);
  const [futures, setFutures] = useState<FuturesBalance | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchBalances() {
      try {
        const spotData = await api.getSpotBalance();
        const futuresData = await api.getFuturesBalance();
        setSpot((spotData as any)?.data ?? (spotData as any) ?? null);
        setFutures((futuresData as any)?.data ?? (futuresData as any) ?? null);
      } catch (err) {
        console.error('❌ Error fetching balances:', err);
      } finally {
        setLoading(false);
      }
    }
    fetchBalances();
  }, []);

  if (loading) return <div className="p-4 bg-gray-900 text-white">⏳ Loading balances...</div>;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      <div className="p-4 bg-gray-900 text-white rounded shadow">
        <h2 className="text-lg font-bold">Spot Balance (USDC)</h2>
        {spot && <p className="text-2xl font-mono">{spot.free} {spot.asset}</p>}
      </div>

      <div className="p-4 bg-gray-900 text-white rounded shadow">
        <h2 className="text-lg font-bold">Futures Balance (USDT-M)</h2>
        {futures && <p className="text-2xl font-mono">{futures.balance} {futures.asset}</p>}
      </div>
    </div>
  );
}
=======
// Auto-generated re-export stub
export { default } from './BalanceCard.tsx';
>>>>>>> Stashed changes
