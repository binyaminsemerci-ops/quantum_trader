// New MUI-based professional dashboard layout
import React, { useMemo } from 'react';
import {
  Container,
  Paper,
  Typography,
  Box,
  Divider,
  Chip,
  Stack,
  useTheme,
  LinearProgress,
} from '@mui/material';
import DashboardIcon from '@mui/icons-material/Dashboard';
import { useDashboardData } from '../hooks/useDashboardData';
import { useSignals } from '../hooks/useSignals';
// Mark React as used to satisfy noUnusedLocals
void React;
import TradeTable from '../components/TradeTable';
import CandlesChart from '../components/CandlesChart';
import PriceChart from '../components/PriceChart';
import EquityChart from '../components/EquityChart';
import TradeLogs from '../components/TradeLogs';
import Watchlist from '../components/Watchlist';
import AITradingMonitor from '../components/AITradingMonitor';

// Minimal live stats fetch (replace later with richer endpoint)
function useLiveStats() {
  const { data } = useDashboardData();
  return useMemo(() => ({
    totalTrades: data?.trades?.length ?? 0,
    samplePnl: (data?.trades || []).reduce((a: number, t: any) => a + (t?.pnl || 0), 0),
  }), [data]);
}

export default function Dashboard(): JSX.Element {
  const theme = useTheme();
  const { data, toast } = useDashboardData();
  const loading = !data; // derive simple loading state until hook provides one
  const { signals } = useSignals({ intervalMs: 7000, limit: 30 });
  const stats = useLiveStats();

  const chartSignals = signals.map(s => ({
    id: s.id || `${s.symbol}-${s.timestamp}`,
    timestamp: s.timestamp,
    score: s.score ?? s.confidence ?? 0,
    direction: s.side === 'sell' ? 'SHORT' : 'LONG'
  }));

  return (
    <Container maxWidth="xl" sx={{ py: 3 }}>
      <Stack direction="row" alignItems="center" spacing={1} mb={2}>
        <DashboardIcon color="primary" fontSize="large" />
        <Typography variant="h4" fontWeight={600}>Quantum Trader Dashboard</Typography>
        <Chip size="small" color="success" label="LIVE" />
        {loading && <LinearProgress sx={{ flex: 1, maxWidth: 200 }} />}
      </Stack>
      {toast && (
        <Paper sx={{ p: 1.5, mb: 2, bgcolor: theme.palette.warning.light }}>
          <Typography variant="body2" fontWeight={500}>{toast.message}</Typography>
        </Paper>
      )}
      <Box sx={{
        display: 'grid',
        gap: 2,
        gridTemplateColumns: {
          xs: '1fr',
          md: 'repeat(12, 1fr)'
        }
      }}>
        {/* Overview */}
        <Paper sx={{ p:2, height:'100%', gridColumn: { xs: '1 / -1', md: 'span 4', lg: 'span 3' } }}>
          <Typography variant="subtitle2" gutterBottom fontWeight={600}>Overview</Typography>
          <Divider sx={{ mb: 1 }} />
            <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(110px,1fr))', gap: 1.2 }}>
              <MiniStat label="Trades" value={stats.totalTrades} />
              <MiniStat label="P&L" value={`$${stats.samplePnl.toFixed(2)}`} />
              <MiniStat label="Signals" value={signals.length} />
              <MiniStat label="Positions" value={0} muted />
            </Box>
        </Paper>
        {/* Candles */}
        <Paper sx={{ p:2, height:'100%', gridColumn: { xs: '1 / -1', md: 'span 8', lg: 'span 9' } }}>
          <SectionHeader title="Market Candles" subtitle="BTCUSDT" />
          <CandlesChart symbol="BTCUSDT" limit={120} />
        </Paper>
        {/* Live Price Action */}
        <Paper sx={{ p:2, height:'100%', gridColumn: { xs: '1 / -1', lg: 'span 8' } }}>
          <SectionHeader title="Live Price Action" subtitle="Signals overlay" />
          <PriceChart signals={chartSignals as any} />
        </Paper>
        {/* Equity Curve */}
        <Paper sx={{ p:2, height:'100%', gridColumn: { xs: '1 / -1', lg: 'span 4' } }}>
          <SectionHeader title="Equity Curve" />
          <EquityChart />
        </Paper>
        {/* Trades */}
        <Paper sx={{ p:2, gridColumn: '1 / -1' }}>
          <SectionHeader title="Trade History" />
          <TradeTable trades={data?.trades} />
        </Paper>
        {/* Activity Logs */}
        <Paper sx={{ p:2, height:'100%', gridColumn: { xs:'1 / -1', md:'span 6' } }}>
          <SectionHeader title="Activity Logs" />
          <TradeLogs />
        </Paper>
        {/* Watchlist */}
        <Paper sx={{ p:2, height:'100%', gridColumn: { xs:'1 / -1', md:'span 6' } }}>
          <SectionHeader title="Watchlist" />
          <Watchlist />
        </Paper>
        {/* AI Trading Monitor */}
        <Paper sx={{ p:2, height:'100%', gridColumn: { xs:'1 / -1', md:'span 6' } }}>
          <SectionHeader title="AI Trading Monitor" />
          <AITradingMonitor />
        </Paper>
      </Box>
    </Container>
  );
}

function MiniStat({ label, value, muted }: { label: string; value: number | string; muted?: boolean }) {
  return (
    <Box sx={{ p: 1.2, borderRadius: 1, bgcolor: muted ? 'action.hover' : 'background.paper', border: theme => `1px solid ${theme.palette.divider}` }}>
      <Typography variant="caption" color="text.secondary">{label}</Typography>
      <Typography variant="subtitle1" fontWeight={600}>{value}</Typography>
    </Box>
  );
}

function SectionHeader({ title, subtitle }: { title: string; subtitle?: string }) {
  return (
    <Box sx={{ mb: 1 }}>
      <Typography variant="subtitle1" fontWeight={600}>{title}</Typography>
      {subtitle && <Typography variant="caption" color="text.secondary">{subtitle}</Typography>}
    </Box>
  );
}

//# sourceMappingURL=Dashboard.js.map