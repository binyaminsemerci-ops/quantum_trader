import { useEffect, useState, useRef } from 'react';
import { formatPercent } from '../utils/format';
import { useDashboardWs } from '../hooks/useDashboardWs';
import axios from 'axios';

interface Signal { id: string; symbol: string; side: 'BUY' | 'SELL'; confidence: number; ts: number }

const symbols = ['BTCUSDC','ETHUSDC','SOLUSDC','ADAUSDC','XRPUSDC'];

function randomSignal(): Signal {
  const sym = symbols[Math.floor(Math.random()*symbols.length)];
  return {
    id: Math.random().toString(36).slice(2),
    symbol: sym,
    side: Math.random() > 0.5 ? 'BUY' : 'SELL',
    confidence: 40 + Math.random()*60,
    ts: Date.now()
  };
}

export default function SignalsWidget({ symbol = 'BTCUSDC' }: { symbol?: string }) {
  const [signals, setSignals] = useState<Signal[]>(() => Array.from({length:5}).map(randomSignal));
  const refreshRef = useRef<number | null>(null);
  const fallbackRef = useRef(false);
  const { data: wsData, status: wsStatus } = useDashboardWs({ enabled: true });

  useEffect(() => {
    let cancelled = false;
    async function fetchSignals() {
      try {
        const res = await axios.get('/api/v1/signals/recent', { params: { symbol, limit: 20 } });
        if (cancelled) return;
        if (Array.isArray(res.data) && res.data.length) {
          const mapped: Signal[] = res.data.map((r: any) => ({
            id: r.id || Math.random().toString(36).slice(2),
            symbol: r.symbol || symbol,
            side: String(r.side || 'BUY').toUpperCase() === 'SELL' ? 'SELL' : 'BUY',
            confidence: Number(r.confidence || 50) * 100, // assuming 0-1 range from backend
            ts: r.timestamp ? Date.parse(r.timestamp) : Date.now(),
          }));
          setSignals(mapped);
          fallbackRef.current = false;
          return;
        }
      } catch {
        if (!fallbackRef.current) {
          fallbackRef.current = true;
        }
      }
      // fallback append one
      setSignals(prev => [randomSignal(), ...prev].slice(0, 25));
    }
    fetchSignals();
    refreshRef.current = window.setInterval(fetchSignals, 6000);
    return () => { cancelled = true; if (refreshRef.current) window.clearInterval(refreshRef.current); };
  }, [symbol]);

  // Integrate websocket trades/logs as pseudo-signals (demo adaptation)
  useEffect(() => {
    if (!wsData || !Array.isArray(wsData.trades) || !wsData.trades.length) return;
    setSignals(prev => {
      const mapped = wsData.trades!.slice(0,5).map((t: any) => ({
        id: 'ws-' + t.id,
        symbol: t.symbol || symbol,
        side: (t.side || 'BUY').toUpperCase() === 'SELL' ? 'SELL' : 'BUY',
        confidence: 70 + Math.random()*25,
        ts: Date.parse(t.timestamp) || Date.now(),
      } as Signal));
      // merge top of list with existing
      const merged = [...mapped, ...prev].slice(0, 25);
      return merged;
    });
  }, [wsData, symbol]);

  return (
    <div className="h-full flex flex-col">
  <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Nyeste signaler {wsStatus === 'open' ? '(live)' : '(mock/fetch)'} </div>
      <div className="flex-1 overflow-auto space-y-1 pr-1">
        {signals.map(s => {
          const buy = s.side === 'BUY';
          const ageSec = Math.max(1, Math.round((Date.now()-s.ts)/1000));
            return (
              <div key={s.id} className="flex items-center justify-between text-[11px] bg-white dark:bg-gray-700 rounded px-2 py-1 border border-gray-200 dark:border-gray-600 font-mono">
                <span className="truncate w-16">{s.symbol.replace('USDC','')}</span>
                <span className={buy? 'text-green-600':'text-red-500'}>{s.side}</span>
                <span className="text-gray-500 dark:text-gray-300">{formatPercent(s.confidence,0)}</span>
                <span className="text-gray-400">{ageSec}s</span>
              </div>
            );
        })}
      </div>
    </div>
  );
}
