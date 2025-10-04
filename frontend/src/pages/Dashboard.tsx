// page-level Dashboard integrates many components via `useDashboardData` hook
// using automatic JSX runtime
import { useEffect, useState } from 'react';
import StatsCard from '../components/StatsCard';
import RiskCards from '../components/RiskCards';
import AnalyticsCards from '../components/AnalyticsCards';
import TradeTable from '../components/TradeTable';
import EquityChart from '../components/EquityChart';
import TradeLogs from '../components/TradeLogs';
import Watchlist from '../components/Watchlist';
import CandlesChart from '../components/CandlesChart';
import PriceChart from '../components/PriceChart';
import AITradingMonitor from '../components/AITradingMonitor';
import CoinPriceMonitor from '../components/CoinPriceMonitor';
import PnLTracker from '../components/PnLTracker';
import AdvancedTechnicalAnalysis from '../components/AdvancedTechnicalAnalysis';
import LoaderOverlay from '../components/LoaderOverlay';
import ErrorBanner from '../components/ErrorBanner';
import Toast from '../components/Toast';
import useDarkMode from '../hooks/useDarkMode';
import { useDashboardData } from '../hooks/useDashboardData';

export default function Dashboard(): JSX.Element {
  const { data, connected, paused, setPaused, fallback, lastUpdated, toast, setToast } = useDashboardData();
  const [darkMode, setDarkMode] = useDarkMode();
  const [signals, setSignals] = useState<any[]>([]);

  // lightweight polling for signals to pass into PriceChart overlay
  useEffect(() => {
    let mounted = true;
    async function fetchSignals() {
      try {
        const base = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';
        const res = await fetch(`${base}/signals/?page=1&page_size=20`);
        if (!res.ok) return;
        const data = await res.json();
        if (mounted) setSignals(data.items || []);
      } catch (e) {
        // ignore
      }
    }
    fetchSignals();
    const id = setInterval(fetchSignals, 5000);
    return () => {
      mounted = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="p-6 space-y-6 dark:bg-gray-900 dark:text-white min-h-screen">
  {/* LoaderOverlay expects no 'show' prop in the typed version; use conditional rendering */}
  {!connected && !fallback && !paused ? <LoaderOverlay /> : null}

      <ErrorBanner show={!connected && !fallback} message="Backend offline – prøver å reconnecte..." />

      {toast && setToast ? (
        <Toast message={toast?.message} type={toast?.type} onClose={() => setToast(null)} />
      ) : null}

      {/* Top Header Bar */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b dark:border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              Quantum Trader Pro
            </h1>
            <span className={`px-3 py-1 rounded-full text-xs font-medium ${
              connected ? 'bg-green-100 text-green-800 dark:bg-green-800 dark:text-green-100' : 
              fallback ? 'bg-yellow-100 text-yellow-800 dark:bg-yellow-800 dark:text-yellow-100' : 
              'bg-red-100 text-red-800 dark:bg-red-800 dark:text-red-100'
            }`}>
              {connected ? '🔴 Live WebSocket' : fallback ? '🟡 REST Polling' : '⚫ Disconnected'}
            </span>
          </div>
          <div className="flex items-center space-x-3">
            {lastUpdated && (
              <span className="text-sm text-gray-500 dark:text-gray-400">
                Updated: {lastUpdated}
              </span>
            )}
            <button 
              onClick={() => setPaused(!paused)} 
              className={`px-4 py-2 rounded-lg font-medium transition-all ${
                paused ? 'bg-green-600 hover:bg-green-700' : 'bg-orange-600 hover:bg-orange-700'
              } text-white shadow-md hover:shadow-lg`}
            >
              {paused ? '▶️ Resume' : '⏸️ Pause'}
            </button>
            <button 
              onClick={() => setDarkMode(!darkMode)} 
              className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg font-medium transition-all shadow-md hover:shadow-lg"
            >
              {darkMode ? '☀️ Light' : '🌙 Dark'}
            </button>
          </div>
        </div>
      </div>

      {/* Main Dashboard Grid */}
      <div className="p-6">
        {/* Top Row - Stats Overview & Controls */}
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 mb-6">
          <div className="lg:col-span-2">
            <StatsCard />
          </div>
          <div>
            <RiskCards />
          </div>
          <div>
            <AnalyticsCards />
          </div>
        </div>

        {/* Second Row - AI Trading & P&L */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
          <div>
            <AITradingMonitor />
          </div>
          <div>
            <PnLTracker />
          </div>
        </div>

        {/* Third Row - Live Prices & Portfolio */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
          <div>
            <CoinPriceMonitor />
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">📊 Portfolio Performance</h3>
            <EquityChart />
          </div>
        </div>

        {/* Fourth Row - Advanced Technical Analysis */}
        <div className="grid grid-cols-1 gap-6 mb-6">
          <div>
            <AdvancedTechnicalAnalysis />
          </div>
        </div>

        {/* Fifth Row - Price Charts */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">📈 Live Price Action</h3>
            <PriceChart signals={signals} />
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">🕯️ Market Candles</h3>
            <CandlesChart symbol="BTCUSDT" limit={50} />
          </div>
        </div>

        {/* Sixth Row - Watchlist & Recent Activity */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 mb-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">👀 Watchlist</h3>
            <Watchlist />
          </div>
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">📋 Recent Activity</h3>
            <TradeLogs />
          </div>
        </div>

        {/* Seventh Row - Active Trades Table */}
        <div className="grid grid-cols-1 gap-6">
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-6 border dark:border-gray-700">
            <h3 className="text-lg font-semibold mb-4 text-gray-800 dark:text-white">🔄 Active Trades</h3>
            <TradeTable trades={data?.trades} />
          </div>
        </div>
      </div>
    </div>
  );
}
