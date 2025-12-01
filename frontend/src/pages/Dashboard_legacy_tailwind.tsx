// Legacy Tailwind-based dashboard preserved for reference/restoration
// This file was created automatically when replacing the dashboard with the new MUI layout.
// You can safely delete it later if not needed.
import React from 'react';
void React;
// ...original legacy dashboard content below...
import { Box, Typography } from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import { useState } from 'react';
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
import CollapsiblePanel from '../components/CollapsiblePanel';
import Toast from '../components/Toast';
import { useDashboardData } from '../hooks/useDashboardData';
import { useSignals } from '../hooks/useSignals';

export default function DashboardLegacy(): JSX.Element {
  const { data, toast } = useDashboardData();
  const { signals: liveSignals } = useSignals({ intervalMs: 7000, limit: 30 });
  const [compact] = useState(() => localStorage.getItem('qt_compact') === '1');
  const chartSignals = liveSignals.map(s => ({
    id: s.id || `${s.symbol}-${s.timestamp}`,
    timestamp: s.timestamp,
    score: s.score ?? s.confidence ?? 0,
    direction: s.side === 'sell' ? 'SHORT' : 'LONG'
  }));
  const showFallback = !data || (Array.isArray(data?.trades) && data.trades.length === 0);
  return (
    <Box sx={{ flexGrow: 1, p: 3 }}>
      <Typography variant="h4" gutterBottom sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
        <DashboardIcon fontSize="large" /> Quantum Trader Dashboard (Legacy)
      </Typography>
      {compact ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-2">
          <div className="min-h-full"><AITradingMonitor /></div>
          <div className="min-h-full"><PnLTracker /></div>
            <div className="min-h-full"><CoinPriceMonitor /></div>
          <div className="sm:col-span-2 lg:col-span-2 xl:col-span-2 min-h-full"><AdvancedTechnicalAnalysis /></div>
          <div className="min-h-full"><CollapsiblePanel title="Equity Curve" icon="📈" variant="compact" defaultExpanded={false}><EquityChart /></CollapsiblePanel></div>
          <div className="min-h-full"><CollapsiblePanel title="Watchlist" icon="👁" variant="compact" defaultExpanded={false}><Watchlist /></CollapsiblePanel></div>
          <div className="min-h-full"><StatsCard /></div>
          <div className="min-h-full"><RiskCards /></div>
          <div className="min-h-full"><AnalyticsCards /></div>
          <div className="min-h-full"><CollapsiblePanel title="Activity Logs" icon="📋" variant="minimal" defaultExpanded={false}><TradeLogs /></CollapsiblePanel></div>
          <div className="sm:col-span-2 lg:col-span-2 xl:col-span-2"><CollapsiblePanel title="Live Price Action" icon="📈" variant="compact" defaultExpanded={false}><PriceChart signals={chartSignals as any} /></CollapsiblePanel></div>
          <div className="sm:col-span-2 lg:col-span-2 xl:col-span-2"><CollapsiblePanel title="Market Candles" icon="🕯️" variant="compact" defaultExpanded={false}><CandlesChart symbol="BTCUSDT" limit={50} /></CollapsiblePanel></div>
          <div className="sm:col-span-2 lg:col-span-4 xl:col-span-4"><CollapsiblePanel title="Trade History" icon="🔄" variant="compact" defaultExpanded={false}><TradeTable trades={data?.trades} /></CollapsiblePanel></div>
        </div>
      ) : (
        <div className="w-full">
          <CollapsiblePanel title="Trade History" icon="🔄" variant="default" defaultExpanded={true}><TradeTable trades={data?.trades} /></CollapsiblePanel>
          <CollapsiblePanel title="Market Candles" icon="🕯️" variant="default" defaultExpanded={true}><CandlesChart symbol="BTCUSDT" limit={100} /></CollapsiblePanel>
          <CollapsiblePanel title="Live Price Action" icon="📈" variant="default" defaultExpanded={true}><PriceChart signals={chartSignals as any} /></CollapsiblePanel>
          <CollapsiblePanel title="Equity Curve" icon="📈" variant="default" defaultExpanded={true}><EquityChart /></CollapsiblePanel>
          <CollapsiblePanel title="Watchlist" icon="👁" variant="default" defaultExpanded={true}><Watchlist /></CollapsiblePanel>
          <CollapsiblePanel title="Activity Logs" icon="📋" variant="default" defaultExpanded={false}><TradeLogs /></CollapsiblePanel>
          <StatsCard />
          <RiskCards />
          <AnalyticsCards />
          <AITradingMonitor />
          <PnLTracker />
          <CoinPriceMonitor />
          <AdvancedTechnicalAnalysis />
        </div>
      )}
      {showFallback && <LoaderOverlay />}
      {toast && <Toast message={toast.message} />}
    </Box>
  );
}
