import StatsCards from "../components/StatsCards.jsx";
import RiskCards from "../components/RiskCards.jsx";
import AnalyticsCards from "../components/AnalyticsCards.jsx";
import TradeTable from "../components/TradeTable.jsx";
import EquityChart from "../components/EquityChart.jsx";
import TradeLogs from "../components/TradeLogs.jsx";
import Watchlist from "../components/Watchlist.jsx";
import CandlesChart from "../components/CandlesChart.jsx";
import LoaderOverlay from "../components/LoaderOverlay.jsx";
import ErrorBanner from "../components/ErrorBanner.jsx";
import Toast from "../components/Toast.jsx";
import useDarkMode from "../hooks/useDarkMode.jsx";
import { useDashboardData } from "../hooks/useDashboardData.jsx";

export default function Dashboard() {
  const {
    connected,
    paused,
    setPaused,
    fallback,
    lastUpdated,
    toast,
    setToast,
  } = useDashboardData();
  const [darkMode, setDarkMode] = useDarkMode();

  return (
    <div className="p-6 space-y-6 dark:bg-gray-900 dark:text-white min-h-screen">
      {/* Overlay loader ved reconnect */}
      <LoaderOverlay show={!connected && !fallback && !paused} />

      {/* Error banner */}
      <ErrorBanner
        show={!connected && !fallback}
        message="Backend offline – prøver å reconnecte..."
      />

      {/* Toast popup */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={() => setToast(null)}
        />
      )}

      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Quantum Trader Dashboard</h1>
        <div className="flex items-center space-x-4">
          <span
            className={`text-sm font-semibold ${
              connected
                ? "text-green-600"
                : fallback
                ? "text-yellow-600"
                : "text-red-600"
            }`}
          >
            {connected
              ? "Live (WebSocket)"
              : fallback
              ? "Fallback (REST Polling)"
              : "Disconnected"}
          </span>
          {lastUpdated && (
            <span className="text-xs text-gray-500 dark:text-gray-400">
              Last updated: {lastUpdated}
            </span>
          )}
          <button
            onClick={() => setPaused(!paused)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg shadow"
          >
            {paused ? "Resume" : "Pause"}
          </button>
          <button
            onClick={() => setDarkMode(!darkMode)}
            className="px-4 py-2 bg-gray-700 text-white rounded-lg shadow"
          >
            {darkMode ? "Light Mode" : "Dark Mode"}
          </button>
        </div>
      </div>

      {/* Dashboard sections */}
      <StatsCards />
      <RiskCards />
      <AnalyticsCards />
      <EquityChart />
      <CandlesChart symbol="BTCUSDT" limit={50} />
      <Watchlist />
      <TradeTable />
      <TradeLogs />
    </div>
  );
}
