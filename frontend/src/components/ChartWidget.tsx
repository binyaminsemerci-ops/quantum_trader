import { useEffect, useState } from 'react';
import { fetchWithFallback } from '../utils/net';
// Derive base URL (reuse Vite env + fallback). Central API wrapper uses '/api',
// but here we need full absolute to call backend directly for legacy endpoint.

interface ChartWidgetProps {
  symbol: string;
  refreshMs?: number;
}

export default function ChartWidget({ symbol, refreshMs = 4000 }: ChartWidgetProps) {
  const [livePrice, setLivePrice] = useState<number>(0);
  const [priceChange, setPriceChange] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    
    async function fetchPrice() {
      try {
  const result = await fetchWithFallback(`/api/v1/portfolio/market-overview`);
        if (!result.ok || !result.data) throw new Error(result.error || 'No data');
        const data: any = result.data;
        if (!data || !Array.isArray(data.symbols)) {
          console.warn('[ChartWidget] market-overview payload missing symbols array', data);
        }
        if (cancelled) return;
        const symbolsArr = data.symbols || [];
        const symbolData = symbolsArr.find((s: any) => s.symbol === symbol) || symbolsArr.find((s: any)=> s.symbol?.startsWith(symbol.replace(/USDC|USDT/,'BTC')));
        if (symbolData) {
          setLivePrice(symbolData.price);
          setPriceChange(symbolData.change24h || 0);
          setLoading(false);
        } else {
          console.warn('[ChartWidget] no symbol match in market overview for', symbol);
          setLoading(false);
        }
      } catch (error) {
        console.error('[ChartWidget] Failed to fetch live price:', error);
        setLoading(false);
      }
    }
    
    fetchPrice();
    const interval = setInterval(fetchPrice, refreshMs);
    
    return () => { 
      cancelled = true; 
      clearInterval(interval);
    };
  }, [refreshMs, symbol]);

  const positive = priceChange >= 0;

  const contentLoading = loading && livePrice === 0;

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-2 text-xs text-gray-500 dark:text-gray-400">
        <span className="font-mono font-semibold">{symbol}</span>
        {contentLoading ? (
          <span className="animate-pulse text-gray-400">Henter...</span>
        ) : (
          <span className={positive ? 'text-green-600 dark:text-green-400' : 'text-red-600 dark:text-red-400'}>
            ${livePrice.toLocaleString()} ({positive ? '+' : ''}{priceChange.toFixed(2)}%)
          </span>
        )}
      </div>
      <div className="flex-1 flex items-center justify-center bg-gray-50 dark:bg-gray-800 rounded">
        <div className="text-center">
          {contentLoading ? (
            <div className="space-y-2">
              <div className="h-6 w-32 mx-auto bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
              <div className="h-4 w-24 mx-auto bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
              <div className="h-3 w-20 mx-auto bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
            </div>
          ) : (
            <>
              <div className="text-2xl font-bold mb-2">${livePrice.toLocaleString()}</div>
              <div className={`text-sm ${positive ? 'text-green-600' : 'text-red-600'}`}>
                {positive ? '↗' : '↘'} {priceChange.toFixed(2)}% (24h)
              </div>
              <div className="text-xs text-gray-500 mt-2">Live Price</div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
