import { useState, useEffect } from 'react';
import { formatCompact } from '../utils/format';
import { Globe, TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react';

interface MarketData {
  marketCap: number;
  volume24h: number;
  dominance: {
    btc: number;
    eth: number;
  };
  fearGreedIndex: number;
  topGainers: {
    symbol: string;
    change: number;
    price: number;
  }[];
  topLosers: {
    symbol: string;
    change: number;
    price: number;
  }[];
}

interface MarketOverviewWidgetProps {
  symbol?: string;
}

const getFearGreedColor = (index: number) => {
  if (index <= 25) return 'text-red-500';
  if (index <= 45) return 'text-orange-500';
  if (index <= 55) return 'text-yellow-500';
  if (index <= 75) return 'text-green-500';
  return 'text-green-600';
};

const getFearGreedLabel = (index: number) => {
  if (index <= 25) return 'Extreme Fear';
  if (index <= 45) return 'Fear';
  if (index <= 55) return 'Neutral';
  if (index <= 75) return 'Greed';
  return 'Extreme Greed';
};

export default function MarketOverviewWidget({ symbol }: MarketOverviewWidgetProps) {
  const [marketData, setMarketData] = useState<MarketData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    
    async function fetchMarketData() {
      try {
        // In development, use relative path so Vite proxy works
        // In production, use VITE_API_BASE_URL
        const isDev = (import.meta as any).env?.DEV;
        const apiBase = isDev ? '' : ((import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000');
        const response = await fetch(`${apiBase}/api/v1/portfolio/market-overview`);
        if (!response.ok) throw new Error('Failed to fetch market data');
        const data = await response.json();
        
        if (cancelled) return;
        
        setMarketData({
          marketCap: data.marketCap || 0,
          volume24h: data.volume24h || 0,
          dominance: {
            btc: data.dominance?.btc || 50,
            eth: data.dominance?.eth || 20
          },
          fearGreedIndex: data.fearGreedIndex || 50,
          topGainers: (data.topGainers || []).map((coin: any) => ({
            symbol: coin.symbol,
            change: coin.change,
            price: coin.price
          })),
          topLosers: (data.topLosers || []).map((coin: any) => ({
            symbol: coin.symbol, 
            change: coin.change,
            price: coin.price
          }))
        });
        setLoading(false);
      } catch (error) {
        console.error('Failed to fetch market data:', error);
        // Fallback to empty data on error
        if (!cancelled) {
          setMarketData({
            marketCap: 0,
            volume24h: 0,
            dominance: { btc: 50, eth: 20 },
            fearGreedIndex: 50,
            topGainers: [],
            topLosers: []
          });
          setLoading(false);
        }
      }
    }
    
    fetchMarketData();
    
    return () => {
      cancelled = true;
    };
  }, [symbol]);

  if (loading || !marketData) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Header */}
      <div className="flex items-center space-x-2">
        <Globe className="w-5 h-5 text-blue-600 dark:text-blue-400" />
        <span className="font-semibold text-gray-900 dark:text-white">Market Overview</span>
      </div>

      {/* Market Stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <DollarSign className="w-4 h-4 text-blue-500" />
            <span className="text-xs text-gray-600 dark:text-gray-400">Market Cap</span>
          </div>
          <div className="text-lg font-bold text-gray-900 dark:text-white">
            {formatCompact(marketData.marketCap)}
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-1">
            <Activity className="w-4 h-4 text-purple-500" />
            <span className="text-xs text-gray-600 dark:text-gray-400">24h Volume</span>
          </div>
          <div className="text-lg font-bold text-gray-900 dark:text-white">
            {formatCompact(marketData.volume24h)}
          </div>
        </div>
      </div>

      {/* Dominance & Fear/Greed */}
      <div className="space-y-3">
        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">Dominance</div>
          <div className="flex items-center justify-between text-sm">
            <span className="text-gray-700 dark:text-gray-300">BTC: {marketData.dominance.btc}%</span>
            <span className="text-gray-700 dark:text-gray-300">ETH: {marketData.dominance.eth}%</span>
          </div>
        </div>

        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-2">Fear & Greed Index</div>
          <div className="flex items-center justify-between">
            <span className={`text-lg font-bold ${getFearGreedColor(marketData.fearGreedIndex)}`}>
              {marketData.fearGreedIndex}
            </span>
            <span className={`text-sm font-semibold ${getFearGreedColor(marketData.fearGreedIndex)}`}>
              {getFearGreedLabel(marketData.fearGreedIndex)}
            </span>
          </div>
        </div>
      </div>

      {/* Top Movers */}
      <div className="flex-1 overflow-hidden">
        <div className="grid grid-cols-2 gap-3 h-full">
          {/* Top Gainers */}
          <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingUp className="w-4 h-4 text-green-500" />
              <span className="text-xs font-semibold text-green-700 dark:text-green-400">Top Gainers</span>
            </div>
            <div className="space-y-2 overflow-y-auto max-h-32">
              {marketData.topGainers.map((coin) => (
                <div key={coin.symbol} className="flex items-center justify-between text-xs">
                  <span className="font-mono text-gray-700 dark:text-gray-300">
                    {coin.symbol.replace('USDC', '')}
                  </span>
                  <div className="text-right">
                    <div className="text-green-600 font-semibold">{coin.change > 0 ? '+' : ''}{coin.change}%</div>
                    <div className="text-gray-500">{coin.price}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Top Losers */}
          <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
            <div className="flex items-center space-x-2 mb-2">
              <TrendingDown className="w-4 h-4 text-red-500" />
              <span className="text-xs font-semibold text-red-700 dark:text-red-400">Top Losers</span>
            </div>
            <div className="space-y-2 overflow-y-auto max-h-32">
              {marketData.topLosers.map((coin) => (
                <div key={coin.symbol} className="flex items-center justify-between text-xs">
                  <span className="font-mono text-gray-700 dark:text-gray-300">
                    {coin.symbol.replace('USDC', '')}
                  </span>
                  <div className="text-right">
                    <div className="text-red-600 font-semibold">{coin.change}%</div>
                    <div className="text-gray-500">{coin.price}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}