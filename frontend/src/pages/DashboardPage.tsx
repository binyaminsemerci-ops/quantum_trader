import { useEffect, useState } from "react";
import PriceChart from "../components/PriceChart";
import SignalFeed from "../components/SignalFeed";
import StressTrendsCard from "../components/StressTrendsCard";
import { fetchRecentPrices } from "../api/prices";
import type { Candle } from "../api/prices";

export default function DashboardPage() {
  const [prices, setPrices] = useState<Candle[] | null>(null);

  useEffect(() => {
    let mounted = true;
    async function load() {
      try {
        const data = await fetchRecentPrices("BTCUSDT", 50);
        if (mounted) setPrices(data);
      } catch (err) {
        // ignore - PriceChart has internal fallback
      }
    }
    load();
    return () => {
      mounted = false;
    };
  }, []);

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-2xl font-bold">Dashboard</h2>
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="md:col-span-2">
          <PriceChart data={prices ?? undefined} />
        </div>
        <div className="md:col-span-1">
          <SignalFeed />
        </div>
      </div>
      <StressTrendsCard />
    </div>
  );
}
