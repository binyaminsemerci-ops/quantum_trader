import { useEffect, useState } from 'react';
import { TrendingUp, TrendingDown, Activity, BarChart3 } from 'lucide-react';
import KpiCard from '../components/analytics/KpiCard';
import EquityChart from '../components/analytics/EquityChart';
import TopList from '../components/analytics/TopList';
import { fetchAnalytics } from '../lib/analyticsApi';

export default function AnalyticsScreen() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadData();
    const interval = setInterval(loadData, 30000); // Refresh every 30s
    return () => clearInterval(interval);
  }, []);

  async function loadData() {
    try {
      console.log('[Analytics] Fetching data...');
      const result = await fetchAnalytics();
      console.log('[Analytics] Data received:', result);
      console.log('[Analytics] Summary:', result?.summary);
      console.log('[Analytics] Has trades:', result?.summary?.trades?.total);
      setData(result);
      setError(null);
    } catch (err: any) {
      console.error('[Analytics] Error loading data:', err);
      console.error('[Analytics] Error stack:', err.stack);
      setError(err.message || 'Failed to load analytics');
    } finally {
      setLoading(false);
    }
  }

  if (loading) {
    return (
      <div className="mx-auto max-w-[1280px] px-4 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="text-slate-500 dark:text-slate-400">Loading analytics...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="mx-auto max-w-[1280px] px-4 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="text-red-500">Error: {error}</div>
        </div>
      </div>
    );
  }

  const { summary, equityCurve, topStrategies, topSymbols } = data || {};

  // Safety check - if no data at all, show loading
  if (!summary) {
    return (
      <div className="mx-auto max-w-[1280px] px-4 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="text-slate-500 dark:text-slate-400">Loading analytics data...</div>
        </div>
      </div>
    );
  }

  // Always show dashboard with available data
  const hasTradeData = summary?.trades?.total > 0;
  const hasBalance = summary?.balance;
  const hasRisk = summary?.risk;

  return (
    <div className="mx-auto max-w-[1280px] px-4 py-6 pb-24">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold">Performance Analytics</h2>
          <p className="text-sm text-slate-500 dark:text-slate-400 mt-1">
            Real-time trading performance metrics
          </p>
        </div>
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
          <div className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="text-sm font-medium text-emerald-600 dark:text-emerald-400">Live</span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        <KpiCard
          title="Total P&L"
          value={hasBalance ? `$${summary.balance.pnl_total.toFixed(2)}` : "$0.00"}
          change={hasBalance ? summary.balance.pnl_pct * 100 : 0}
          changeLabel="Total Return"
          trend={hasBalance && summary.balance.pnl_total > 0 ? 'up' : hasBalance && summary.balance.pnl_total < 0 ? 'down' : 'neutral'}
          icon={<TrendingUp className="w-5 h-5" />}
        />
        <KpiCard
          title="Win Rate"
          value={hasTradeData ? `${(summary.trades.win_rate * 100).toFixed(1)}%` : "0.0%"}
          subtitle={hasTradeData ? `${summary.trades.total} trades` : "0 trades"}
          trend={hasTradeData && summary.trades.win_rate > 0.5 ? 'up' : 'neutral'}
          icon={<Activity className="w-5 h-5" />}
        />
        <KpiCard
          title="Max Drawdown"
          value={hasRisk ? `${(summary.risk.max_drawdown * 100).toFixed(2)}%` : "0.00%"}
          trend={hasRisk && summary.risk.max_drawdown < -0.1 ? 'down' : 'neutral'}
          icon={<TrendingDown className="w-5 h-5" />}
        />
        <KpiCard
          title="Total Trades"
          value={hasTradeData ? summary.trades.total.toString() : "0"}
          subtitle={hasBalance ? `Balance: $${summary.balance.current.toFixed(2)}` : "Starting balance: $10,000"}
          trend="neutral"
          icon={<BarChart3 className="w-5 h-5" />}
        />
      </div>

      {/* Equity Chart & Top Strategies */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        <div className="lg:col-span-2">
          <EquityChart data={equityCurve || []} />
        </div>
        <div>
          <TopList
            title="Top Strategies"
            items={(topStrategies || []).map((s: any) => ({
              name: s.strategy_id,
              value: s.total_pnl,
              subtitle: `${s.total_trades} trades`,
              trend: s.total_pnl > 0 ? 'up' : 'down'
            }))}
            layout="vertical"
          />
        </div>
      </div>

      {/* Top Symbols */}
      <TopList
        title="Top Symbols"
        items={(topSymbols || []).map((s: any) => ({
          name: s.symbol,
          value: s.total_pnl,
          subtitle: `${s.total_trades} trades`,
          trend: s.total_pnl > 0 ? 'up' : 'down'
        }))}
        layout="horizontal"
      />
    </div>
  );
}
