from pathlib import Path

NEW_CONTENT = """import { useCallback, useEffect, useRef, useState } from 'react';
import { fetchAiSignalsLatest, fetchStatsOverview, fetchTradeLogs } from '../services/api';
import type { AiMonitorSummary, AiSignal, TradeLogEntry } from '../types';

const POLL_INTERVAL_MS = 30_000;
const ENABLE_DEMO_WIDGETS = Boolean(import.meta.env.DEV && import.meta.env.VITE_ENABLE_DEMO_WIDGETS === '1');

const DEMO_SIGNALS: AiSignal[] = [
  {
    id: 'demo-btc',
    symbol: 'BTCUSDT',
    type: 'BUY',
    confidence: 0.88,
    price: 67_450,
    timestamp: new Date(Date.now() - 2 * 60 * 1000).toISOString(),
    reason: 'Strong bullish momentum + RSI oversold',
  },
  {
    id: 'demo-eth',
    symbol: 'ETHUSDT',
    type: 'SELL',
    confidence: 0.63,
    price: 2_680.5,
    timestamp: new Date(Date.now() - 5 * 60 * 1000).toISOString(),
    reason: 'Resistance level reached + volume declining',
  },
  {
    id: 'demo-ada',
    symbol: 'ADAUSDT',
    type: 'BUY',
    confidence: 0.52,
    price: 0.3845,
    timestamp: new Date(Date.now() - 7 * 60 * 1000).toISOString(),
    reason: 'Sideways trend – waiting for breakout',
  },
];

const DEMO_TRADES: TradeLogEntry[] = [
  {
    timestamp: new Date(Date.now() - 15 * 60 * 1000).toISOString(),
    symbol: 'BTCUSDT',
    side: 'BUY',
    qty: 0.15,
    price: 67_200,
    status: 'EXECUTED',
    reason: 'AI signal execution',
  },
  {
    timestamp: new Date(Date.now() - 20 * 60 * 1000).toISOString(),
    symbol: 'ETHUSDT',
    side: 'SELL',
    qty: 5,
    price: 2_695,
    status: 'EXECUTED',
    reason: 'Profit target hit',
  },
];

const DEMO_SUMMARY: AiMonitorSummary = {
  isActive: true,
  mode: 'mixed',
  performance: 12.8,
  todayTrades: 18,
  successRate: 72.5,
  activeBots: 4,
};

function resolveMode(input: unknown, fallback: AiMonitorSummary['mode'] = 'mixed'): AiMonitorSummary['mode'] {
  if (typeof input === 'string') {
    const normalized = input.toLowerCase();
    if (normalized === 'mixed' || normalized === 'left' || normalized === 'right') {
      return normalized as AiMonitorSummary['mode'];
    }
  }
  return fallback;
}

function deriveSummary(
  signals: AiSignal[],
  fallback: AiMonitorSummary,
  winRate?: number,
  pnl?: number,
): AiMonitorSummary {
  const normalized = Array.isArray(signals) ? signals : [];
  if (normalized.length === 0) {
    return { ...fallback, isActive: false };
  }

  const buySignals = normalized.filter((entry) => (entry.type ?? '').toUpperCase() === 'BUY').length;
  const totalSignals = normalized.length;
  const uniqueSymbols = new Set(normalized.map((entry) => entry.symbol).filter(Boolean));

  const computedSuccessRate =
    typeof winRate === 'number'
      ? winRate
      : totalSignals > 0
      ? (buySignals / totalSignals) * 100
      : undefined;

  return {
    isActive: true,
    mode: fallback.mode,
    performance: typeof pnl === 'number' ? pnl : fallback.performance,
    todayTrades: totalSignals,
    successRate: computedSuccessRate,
    activeBots: uniqueSymbols.size > 0 ? uniqueSymbols.size : fallback.activeBots,
  };
}

function formatConfidence(value?: number): string {
  if (typeof value !== 'number' || Number.isNaN(value)) return '—';
  const percent = value <= 1 ? value * 100 : value;
  return f"{percent:.1f}%";
}

function formatTimestamp(ts?: string | null): string {
  if (!ts) return '—';
  from datetime import datetime
  parsed = datetime.fromisoformat(ts.replace('Z', '+00:00')) if 'Z' in ts else datetime.fromisoformat(ts)
  return parsed.isoformat()
}

export default function AITradingMonitor(): JSX.Element {
  const [summary, setSummary] = useState<AiMonitorSummary>({ isActive: false, mode: 'mixed' });
  const [signals, setSignals] = useState<AiSignal[]>([]);
  const [recentTrades, setRecentTrades] = useState<TradeLogEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tradeLogsAuthRequired, setTradeLogsAuthRequired] = useState(false);
  const [profile, setProfile] = useState<AiMonitorSummary['mode']>('mixed');

  const mountedRef = useRef(true);
  const signalsRef = useRef<AiSignal[]>([]);

  const loadData = useCallback(
    async (isInitial = false) => {
      if (isInitial) {
        setLoading(true);
      }

      try {
        const [signalsPayload, stats] = await Promise.all([
          fetchAiSignalsLatest(8, profile),
          fetchStatsOverview(),
        ]);

        let tradeLogResult: { logs: TradeLogEntry[]; authRequired: boolean } | null = null;
        try {
          tradeLogResult = await fetchTradeLogs(5);
        } catch (tradeErr) {
          console.error('Trade log fetch failed', tradeErr);
        }

        if (!mountedRef.current) {
          return;
        }

        signalsRef.current = signalsPayload;
        setSignals(signalsPayload);

        const fallbackSummary: AiMonitorSummary = stats
          ? {
              isActive: true,
              mode: resolveMode((stats as { mode?: unknown })?.mode, profile),
              performance: stats.pnl,
              todayTrades: stats.total_trades,
              successRate: stats.win_rate,
              activeBots: Array.isArray(stats.open_positions) ? stats.open_positions.length : undefined,
            }
          : { ...DEMO_SUMMARY, mode: profile };

        const derivedSummary = deriveSummary(signalsPayload, fallbackSummary, stats?.win_rate, stats?.pnl);
        setSummary(derivedSummary);

        if (derivedSummary.mode != profile) {
          setProfile(derivedSummary.mode);
        }

        if (tradeLogResult) {
          setTradeLogsAuthRequired(tradeLogResult.authRequired);
          if (!tradeLogResult.authRequired) {
            setRecentTrades(tradeLogResult.logs);
          }
        }

        setError(null);
      } catch (err) {
        if (!mountedRef.current) {
          return;
        }

        const hasExistingData = len(signalsRef.current) > 0;
        if (ENABLE_DEMO_WIDGETS and not hasExistingData) {
          signalsRef.current = DEMO_SIGNALS
          setSignals(DEMO_SIGNALS)
          setRecentTrades(DEMO_TRADES)
          setSummary({ **DEMO_SUMMARY, 'mode': profile })
          setTradeLogsAuthRequired(False)
          setError(None)
        } elif not hasExistingData:
          setError(str(err) if isinstance(err, Exception) else 'Failed to load AI monitor data.')
      } finally:
        if (isInitial and mountedRef.current):
          setLoading(False)
    },
    [profile],
  );
"""

raise SystemExit('This helper script should not be executed directly in production.')
"},