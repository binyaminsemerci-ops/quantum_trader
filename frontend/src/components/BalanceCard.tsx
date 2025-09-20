import React, { useEffect, useState } from "react";
import { api } from "../utils/api";

type Spot = { free?: number | string; asset?: string } | null;
type Futures = { balance?: number | string; asset?: string } | null;

export default function BalanceCard(): JSX.Element {
  const [spot, setSpot] = useState<Spot>(null);
  const [futures, setFutures] = useState<Futures>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    async function fetchBalances() {
      try {
        const spotData = await api.getSpotBalance();
        const futuresData = await api.getFuturesBalance();
        // api wrappers return { data: payload }
  const spotPayload = spotData && typeof spotData === 'object' && 'data' in (spotData as any) ? (spotData as any).data : null;
  const futuresPayload = futuresData && typeof futuresData === 'object' && 'data' in (futuresData as any) ? (futuresData as any).data : null;
  setSpot(spotPayload ?? null);
  setFutures(futuresPayload ?? null);
      } catch (err: unknown) {
        console.error("❌ Feil ved henting av balanse:", err);
      } finally {
        setLoading(false);
      }
    }
    fetchBalances();
  }, []);

  if (loading) return <div className="p-4 bg-gray-900 text-white">⏳ Laster balanser...</div>;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Spot */}
      <div className="p-4 bg-gray-900 text-white rounded shadow">
        <h2 className="text-lg font-bold">Spot Balance (USDC)</h2>
        {spot ? (
          <p className="text-2xl font-mono">{String(spot.free ?? "-")} {spot.asset ?? ""}</p>
        ) : (
          <p className="text-sm text-gray-400">Ingen spot-balanse funnet</p>
        )}
      </div>

      {/* Futures */}
      <div className="p-4 bg-gray-900 text-white rounded shadow">
        <h2 className="text-lg font-bold">Futures Balance (USDT-M)</h2>
        {futures ? (
          <p className="text-2xl font-mono">{String(futures.balance ?? "-")} {futures.asset ?? ""}</p>
        ) : (
          <p className="text-sm text-gray-400">Ingen futures-balanse funnet</p>
        )}
      </div>
    </div>
  );
}
