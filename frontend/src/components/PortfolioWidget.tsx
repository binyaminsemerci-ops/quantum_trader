import { useState, useEffect } from 'react';
import { fetchWithFallback } from '../utils/net';
import { formatCurrency, formatPercent } from '../utils/format';
import { TrendingUp, TrendingDown, DollarSign, Wallet } from 'lucide-react';

interface PortfolioData {
  totalValue: number;
  totalPnL: number;
  totalPnLPercent: number;
  positions: {
    symbol: string;
    amount: number;
    value: number;
    pnl: number;
    pnlPercent: number;
  }[];
}

interface PortfolioWidgetProps {
  symbol?: string;
}

export default function PortfolioWidget({ symbol }: PortfolioWidgetProps) {
  const [portfolio, setPortfolio] = useState<PortfolioData>({
    totalValue: 0,
    totalPnL: 0,
    totalPnLPercent: 0,
    positions: []
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    let interval: any;
    async function fetchPortfolioData() {
      try {
  const result = await fetchWithFallback(`/api/v1/portfolio`);
  if (!result.ok || !result.data) throw new Error(result.error || 'Portfolio fetch failed');
  const data: any = result.data;
        if (cancelled) return;
        const positionsRaw = (data.positions || []);
        if (!Array.isArray(positionsRaw)) {
          console.warn('[PortfolioWidget] positions field not array', data.positions);
        }
        setPortfolio({
          totalValue: data.totalValue || 0,
            totalPnL: data.totalPnL || 0,
            totalPnLPercent: data.dayPnL && data.totalValue ? (data.dayPnL / data.totalValue) * 100 : 0,
            positions: positionsRaw.map((pos: any) => ({
              symbol: pos.symbol,
              amount: pos.quantity || 0,
              value: pos.marketValue || 0,
              pnl: pos.unrealizedPnL || 0,
              pnlPercent: pos.avgPrice && pos.currentPrice ? ((pos.currentPrice - pos.avgPrice) / pos.avgPrice) * 100 : 0
            }))
        });
        setLoading(false);
      } catch (error) {
        console.error('[PortfolioWidget] Failed to fetch portfolio data:', error);
        if (!cancelled) setLoading(false);
      }
    }
    fetchPortfolioData();
    interval = setInterval(fetchPortfolioData, 8000);
    return () => { cancelled = true; clearInterval(interval); };
  }, [symbol]);

  const showSkeleton = loading && portfolio.totalValue === 0 && portfolio.positions.length === 0;

  const isPositive = portfolio.totalPnL >= 0;

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Portfolio Summary */}
  <div className="bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg p-4 min-h-[110px] flex flex-col">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <Wallet className="w-5 h-5 text-blue-600 dark:text-blue-400" />
            <span className="font-semibold text-gray-900 dark:text-white">Total Portfolio</span>
          </div>
          {isPositive ? (
            <TrendingUp className="w-5 h-5 text-green-500" />
          ) : (
            <TrendingDown className="w-5 h-5 text-red-500" />
          )}
        </div>
        
        <div className="space-y-1">
          {showSkeleton ? (
            <div className="w-full space-y-2">
              <div className="h-6 w-40 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
              <div className="h-4 w-24 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
            </div>
          ) : (
            <div className="flex items-center space-x-2">
              <DollarSign className="w-4 h-4 text-gray-500" />
              <span className="text-2xl font-bold text-gray-900 dark:text-white">
                {formatCurrency(portfolio.totalValue, 'USDC')}
              </span>
            </div>
          )}
          
          {!showSkeleton && (
            <div className="flex items-center space-x-2 text-sm">
              <span className={`font-semibold ${isPositive ? 'text-green-600' : 'text-red-600'}`}>
                {isPositive ? '' : '-'}{formatCurrency(Math.abs(portfolio.totalPnL), 'USDC')} {formatPercent(portfolio.totalPnLPercent)}
              </span>
              <span className="text-gray-500 dark:text-gray-400">24h</span>
            </div>
          )}
        </div>
      </div>

      {/* Positions */}
      <div className="flex-1 overflow-hidden">
        <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">Positions</h4>
        <div className="space-y-2 overflow-y-auto max-h-64">
          {portfolio.positions.map((position) => (
            <div 
              key={position.symbol}
              className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600"
            >
              <div className="flex items-center justify-between mb-1">
                <span className="font-mono text-sm font-semibold text-gray-900 dark:text-white">
                  {position.symbol.replace('USDC', '/USDC')}
                </span>
                <span className="text-xs text-gray-500 dark:text-gray-400">
                  {position.amount} units
                </span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-900 dark:text-white">
                  {formatCurrency(position.value, 'USDC')}
                </span>
                <span className={`text-sm font-semibold ${
                  position.pnl >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {position.pnl >= 0 ? '' : '-'}{formatCurrency(Math.abs(position.pnl), 'USDC')}
                  <span className="text-xs ml-1">({formatPercent(position.pnlPercent)})</span>
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}