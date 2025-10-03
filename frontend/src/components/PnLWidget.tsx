import { useState, useEffect } from 'react';
import { fetchWithFallback } from '../utils/net';
import { formatCurrency, formatPercent } from '../utils/format';
import { TrendingUp, TrendingDown, DollarSign, Target } from 'lucide-react';

interface PnLData {
  dailyPnL: number;
  weeklyPnL: number;
  monthlyPnL: number;
  totalPnL: number;
  winRate: number;
  avgWin: number;
  avgLoss: number;
  sharpeRatio: number;
  maxDrawdown: number;
  recentTrades: {
    symbol: string;
    side: 'BUY' | 'SELL';
    pnl: number;
    timestamp: string;
  }[];
}

interface PnLWidgetProps {
  symbol?: string;
}

export default function PnLWidget({ symbol }: PnLWidgetProps) {
  const [pnlData, setPnlData] = useState<PnLData | null>(null);
  const [loading, setLoading] = useState(true);
  const [timeframe, setTimeframe] = useState<'1D' | '1W' | '1M' | 'ALL'>('1D');

  useEffect(() => {
    let cancelled = false;
    
    async function fetchPnLData() {
      try {
        const result = await fetchWithFallback(`/api/v1/portfolio/pnl`);
        if (!result.ok || !result.data) throw new Error(result.error || 'PnL fetch failed');
        const data: any = result.data;
        if (data.totalPnL === undefined) {
          console.warn('[PnLWidget] unexpected payload structure', data);
        }
        
        if (cancelled) return;
        
        setPnlData({
          dailyPnL: data.dayPnL || 0,
          weeklyPnL: data.weekPnL || 0,
          monthlyPnL: data.monthPnL || 0,
          totalPnL: data.totalPnL || 0,
          winRate: data.winRate || 0,
          avgWin: data.avgWin || 0,
          avgLoss: data.avgLoss || 0,
          sharpeRatio: data.sharpeRatio || 0,
          maxDrawdown: data.maxDrawdown || 0,
          recentTrades: (data.recentTrades || []).map((trade: any) => ({
            symbol: trade.symbol,
            side: trade.side,
            pnl: trade.pnl,
            timestamp: trade.timestamp
          }))
        });
        setLoading(false);
      } catch (error) {
  console.error('[PnLWidget] Failed to fetch P&L data:', error);
        // Fallback to empty data on error
        if (!cancelled) {
          setPnlData({
            dailyPnL: 0,
            weeklyPnL: 0,
            monthlyPnL: 0,
            totalPnL: 0,
            winRate: 0,
            avgWin: 0,
            avgLoss: 0,
            sharpeRatio: 0,
            maxDrawdown: 0,
            recentTrades: []
          });
          setLoading(false);
        }
      }
    }
    
    fetchPnLData();
    const interval = setInterval(fetchPnLData, 10000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [symbol]);

  const showSkeleton = (loading || !pnlData);

  const getCurrentPnL = () => {
    if (!pnlData) return 0;
    switch (timeframe) {
      case '1D': return pnlData.dailyPnL;
      case '1W': return pnlData.weeklyPnL;
      case '1M': return pnlData.monthlyPnL;
      case 'ALL': return pnlData.totalPnL;
      default: return pnlData.dailyPnL;
    }
  };

  const currentPnL = getCurrentPnL();
  const isPositive = currentPnL >= 0;

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* P&L Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Target className="w-5 h-5 text-green-600 dark:text-green-400" />
          <span className="font-semibold text-gray-900 dark:text-white">P&L Tracker</span>
        </div>
        
        {/* Timeframe Selector */}
        <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          {(['1D', '1W', '1M', 'ALL'] as const).map((tf) => (
            <button
              key={tf}
              onClick={() => setTimeframe(tf)}
              className={`px-2 py-1 text-xs rounded transition-colors ${
                timeframe === tf
                  ? 'bg-blue-500 text-white'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      {/* Current P&L */}
      <div className={`rounded-lg p-4 min-h-[120px] ${
        !showSkeleton && isPositive 
          ? 'bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20'
          : !showSkeleton
            ? 'bg-gradient-to-r from-red-50 to-pink-50 dark:from-red-900/20 dark:to-pink-900/20'
            : 'bg-gray-100 dark:bg-gray-800'
      }`}>
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-600 dark:text-gray-400">
            {timeframe} P&L
          </span>
          {isPositive ? (
            <TrendingUp className="w-5 h-5 text-green-500" />
          ) : (
            <TrendingDown className="w-5 h-5 text-red-500" />
          )}
        </div>
        
        {showSkeleton ? (
          <div className="space-y-2">
            <div className="h-6 w-32 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
            <div className="h-4 w-20 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
          </div>
        ) : (
          <div className="flex items-center space-x-2">
            <DollarSign className="w-5 h-5 text-gray-500" />
            <span className={`text-2xl font-bold ${
              isPositive ? 'text-green-600' : 'text-red-600'
            }`}>
              {isPositive ? '' : '-'}{formatCurrency(Math.abs(currentPnL), 'USDC')}
            </span>
          </div>
        )}
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 gap-3 flex-1">
        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Win Rate</div>
          {showSkeleton ? <div className="h-5 w-12 bg-gray-200 dark:bg-gray-600 rounded animate-pulse" /> : <div className="text-lg font-semibold text-green-600">{formatPercent(pnlData.winRate, 1)}</div>}
        </div>

        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Sharpe Ratio</div>
          {showSkeleton ? <div className="h-5 w-10 bg-gray-200 dark:bg-gray-600 rounded animate-pulse" /> : <div className="text-lg font-semibold text-blue-600">{pnlData.sharpeRatio.toFixed(2)}</div>}
        </div>

        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Avg Win</div>
          {showSkeleton ? <div className="h-5 w-16 bg-gray-200 dark:bg-gray-600 rounded animate-pulse" /> : <div className="text-lg font-semibold text-green-600">{formatCurrency(pnlData.avgWin, 'USDC')}</div>}
        </div>

        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Max DD</div>
          {showSkeleton ? <div className="h-5 w-10 bg-gray-200 dark:bg-gray-600 rounded animate-pulse" /> : <div className="text-lg font-semibold text-red-600">{formatPercent(pnlData.maxDrawdown, 1)}</div>}
        </div>
      </div>

      {/* Recent Trades */}
      <div className="flex-1 overflow-hidden">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Recent Trades</h4>
        <div className="space-y-1 overflow-y-auto max-h-32">
          {showSkeleton ? (
            Array.from({length:4}).map((_,i)=>(<div key={i} className="h-6 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />))
          ) : pnlData.recentTrades.map((trade, index) => (
            <div 
              key={index}
              className="flex items-center justify-between py-1 px-2 bg-gray-50 dark:bg-gray-800 rounded text-xs"
            >
              <div className="flex items-center space-x-2">
                <span className="font-mono">{trade.symbol.replace('USDC', '')}</span>
                <span className={`px-1 rounded text-xs ${
                  trade.side === 'BUY' 
                    ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' 
                    : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
                }`}>
                  {trade.side}
                </span>
              </div>
              <span className={`font-semibold ${
                trade.pnl >= 0 ? 'text-green-600' : 'text-red-600'
              }`}>
                {trade.pnl >= 0 ? '' : '-'}{formatCurrency(Math.abs(trade.pnl), 'USDC')}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}