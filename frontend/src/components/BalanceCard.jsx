import React, { useEffect, useState } from "react";
import { api } from "../utils/api";

export default function BalanceCard() {
  const [spot, setSpot] = useState(null);
  const [futures, setFutures] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function fetchBalances() {
      try {
        const spotData = await api.getSpotBalance();
        const futuresData = await api.getFuturesBalance();
        setSpot(spotData);
        setFutures(futuresData);
      } catch (err) {
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
        {spot && <p className="text-2xl font-mono">{spot.free} {spot.asset}</p>}
      </div>

      {/* Futures */}
      <div className="p-4 bg-gray-900 text-white rounded shadow">
        <h2 className="text-lg font-bold">Futures Balance (USDT-M)</h2>
        {futures && <p className="text-2xl font-mono">{futures.balance} {futures.asset}</p>}
      </div>
    </div>
  );
}
