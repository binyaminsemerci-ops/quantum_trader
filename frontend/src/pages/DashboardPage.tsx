import { useEffect, useState } from "react";
import PriceChart from "../components/PriceChart";
import SignalFeed from "../components/SignalFeed";
import StressTrendsCard from "../components/StressTrendsCard";
import { fetchRecentPrices } from "../api/prices";
import { fetchSignals, type Signal } from "../api/signals";
import type { Candle } from "../api/prices";

const DEFAULT_SYMBOL = "BTCUSDT";
const DEFAULT_LIMIT = 50;

export default function DashboardPage() {
  const [prices, setPrices] = useState<Candle[] | null>(null);
  const [priceSource, setPriceSource] = useState<'live' | 'demo' | null>(null);
  const [signals, setSignals] = useState<Signal[] | null>(null);
  const [priceError, setPriceError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function loadPrices() {
      try {
        const result = await fetchRecentPrices(DEFAULT_SYMBOL, DEFAULT_LIMIT);
        if (cancelled) return;
        setPrices(result.candles);
        setPriceSource(result.source);
        setPriceError(null);
      } catch (err) {
        if (cancelled) return;
        console.warn("price refresh failed", err);
        setPriceError('Failed to load latest prices');
      }
    }
    loadPrices();
    const id = window.setInterval(loadPrices, 15_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    async function loadSignals() {
      try {
        const { items } = await fetchSignals({ limit: 40, symbol: DEFAULT_SYMBOL });
        if (!cancelled) setSignals(items);
      } catch (err) {
        if (!cancelled) {
          console.warn("signal overlay refresh failed", err);
          setSignals([]);
        }
      }
    }
    loadSignals();
    const id = window.setInterval(loadSignals, 10_000);
    return () => {
      cancelled = true;
      window.clearInterval(id);
    };
  }, []);

  return (
    <div className="p-4 space-y-4">
      <h2 className="text-2xl font-bold">Dashboard</h2>
      {priceError && (
        <div className="text-xs text-amber-600" role="alert">
          {priceError}
        </div>
      )}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <div className="md:col-span-2">
          <PriceChart data={prices ?? undefined} signals={signals ?? undefined} source={priceSource ?? undefined} />
        </div>
        <div className="md:col-span-1">
          <SignalFeed symbol={DEFAULT_SYMBOL} />
        </div>
      </div>
      <StressTrendsCard />
    </div>
  );
}
