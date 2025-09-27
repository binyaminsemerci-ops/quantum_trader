// page-level Dashboard integrates many components via `useDashboardData` hook
// using automatic JSX runtime
import StatsCard from '../components/StatsCard';
import RiskCards from '../components/RiskCards';
import AnalyticsCards from '../components/AnalyticsCards';
import TradeTable from '../components/TradeTable';
import EquityChart from '../components/EquityChart';
import TradeLogs from '../components/TradeLogs';
import Watchlist from '../components/Watchlist';
import CandlesChart from '../components/CandlesChart';
import LoaderOverlay from '../components/LoaderOverlay';
import ErrorBanner from '../components/ErrorBanner';
import Toast from '../components/Toast';
import useDarkMode from '../hooks/useDarkMode';
import { useDashboardData } from '../hooks/useDashboardData';

export default function Dashboard(): JSX.Element {
  const { data, connected, paused, setPaused, fallback, lastUpdated, toast, setToast } = useDashboardData();
  const [darkMode, setDarkMode] = useDarkMode();

  return (
    <div className="p-6 space-y-6 dark:bg-gray-900 dark:text-white min-h-screen">
  {/* LoaderOverlay expects no 'show' prop in the typed version; use conditional rendering */}
  {!connected && !fallback && !paused ? <LoaderOverlay /> : null}

      <ErrorBanner show={!connected && !fallback} message="Backend offline – prøver å reconnecte..." />

      {toast && setToast ? (
        <Toast message={toast?.message} type={toast?.type} onClose={() => setToast(null)} />
      ) : null}

      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Quantum Trader Dashboard</h1>
        <div className="flex items-center space-x-4">
          <span className={`text-sm font-semibold ${connected ? 'text-green-600' : fallback ? 'text-yellow-600' : 'text-red-600'}`}>
            {connected ? 'Live (WebSocket)' : fallback ? 'Fallback (REST Polling)' : 'Disconnected'}
          </span>
          {lastUpdated && <span className="text-xs text-gray-500 dark:text-gray-400">Last updated: {lastUpdated}</span>}
          <button onClick={() => setPaused(!paused)} className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow">{paused ? 'Resume' : 'Pause'}</button>
          <button onClick={() => setDarkMode(!darkMode)} className="px-4 py-2 bg-gray-700 text-white rounded-lg shadow">{darkMode ? 'Light Mode' : 'Dark Mode'}</button>
        </div>
      </div>

  <StatsCard />
      <RiskCards />
      <AnalyticsCards />
      <EquityChart />
      <CandlesChart symbol="BTCUSDT" limit={50} />
      <Watchlist />
  <TradeTable trades={data?.trades} />
      <TradeLogs />
    </div>
  );
}
