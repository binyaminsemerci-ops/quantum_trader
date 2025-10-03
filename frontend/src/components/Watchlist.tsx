import { useEffect, useState } from 'react';
import { useDashboardData } from '../hooks/useDashboardData';
import Toast from './Toast';

type PriceRow = {
  symbol: string;
  price?: number;
  change24h?: number;
  volume24h?: number;
  sparkline?: number[];
  error?: string;
};

function Sparkline({ data }: { data: number[] | undefined }) {
  if (!data || data.length === 0) return <div className="text-xs text-gray-500">-</div>;
  const w = 120;
  const h = 28;
  const min = Math.min(...data);
  const max = Math.max(...data);
  const scale = (v: number) => ((v - min) / (max - min || 1)) * h;
  const points = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - scale(v)}`).join(' ');
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`}>
      <polyline
        fill="none"
        stroke="#4f46e5"
        strokeWidth={1.5}
        points={points}
      />
    </svg>
  );
}

export default function Watchlist(): JSX.Element {
  const { data } = useDashboardData();
  const symbolsPnl = data?.stats?.pnl_per_symbol || {};
  const [rows, setRows] = useState<PriceRow[]>([]);
  const watched = Object.keys(symbolsPnl).length ? Object.keys(symbolsPnl) : ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'];

  useEffect(() => {
    let mounted = true;
    let ws: WebSocket | null = null;

    const connectWs = () => {
      try {
        const q = watched.join(',');
        const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || '';
        if (apiBase) {
          const wsBase = apiBase.replace(/^http/, 'ws').replace(/\/$/, '');
          ws = new WebSocket(`${wsBase}/watchlist/ws/watchlist?symbols=${encodeURIComponent(q)}&limit=24`);
        } else {
          // When running via Vite dev server, use the /api proxy so requests reach the backend
          ws = new WebSocket(`${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/watchlist/ws/watchlist?symbols=${encodeURIComponent(q)}&limit=24`);
        }
        ws.onmessage = (ev) => {
          if (!mounted) return;
          try {
            const json = JSON.parse(ev.data);
            setRows(json.map((r: any) => ({
              symbol: r.symbol,
              price: r.price,
              change24h: r.change24h,
              volume24h: r.volume24h,
              sparkline: r.sparkline,
              error: r.error,
            })));
          } catch (e) {
            // ignore parse errors
          }
        };
        ws.onclose = () => {
          if (!mounted) return;
          // fallback to polling after short delay
          setTimeout(startPolling, 1000);
        };
      } catch (e) {
        startPolling();
      }
    };

    let pollId: number | null = null;
    async function startPolling() {
      const q = watched.join(',');
      try {
        const res = await fetch(`/api/watchlist/prices?symbols=${encodeURIComponent(q)}&limit=24`);
        const json = await res.json();
        if (!mounted) return;
        setRows(json.map((r: any) => ({
          symbol: r.symbol,
          price: r.price,
          change24h: r.change24h,
          volume24h: r.volume24h,
          sparkline: r.sparkline,
          error: r.error,
        })));
      } catch (e) {
        // ignore
      }
      if (pollId) window.clearTimeout(pollId);
      pollId = window.setTimeout(startPolling, 5000) as unknown as number;
    }

    // Try WS first, fallback to polling
    connectWs();

    return () => {
      mounted = false;
      if (ws) {
        try { ws.close(); } catch (e) {}
      }
    };
  }, [JSON.stringify(watched)]);

  return (
    <div className="bg-white dark:bg-gray-800 p-4 shadow rounded-lg mt-4">
      <h3 className="font-bold mb-2">Watchlist</h3>
      <table className="min-w-full">
        <thead>
          <tr>
            <th className="text-left">Symbol</th>
            <th className="text-left">Price</th>
            <th className="text-left">24h</th>
            <th className="text-left">Volume</th>
            <th className="text-left">Sparkline</th>
            <th className="text-left">PnL</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.symbol} className="align-top">
              <td className="font-mono">{r.symbol}</td>
              <td>{r.price ? r.price.toFixed(2) : '-'}</td>
              <td className={r.change24h && r.change24h >= 0 ? 'text-green-600' : 'text-red-600'}>
                {r.change24h ? `${(r.change24h * 100).toFixed(2)}%` : '-'}
              </td>
              <td>{r.volume24h ? r.volume24h.toFixed(2) : '-'}</td>
              <td><Sparkline data={r.sparkline} /></td>
              <td className={Number(symbolsPnl[r.symbol] || 0) >= 0 ? 'text-green-600' : 'text-red-600'}>
                {String(symbolsPnl[r.symbol] ?? '-')}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4">
        <h4 className="font-semibold">Manage Watchlist</h4>
        <ManageWatchlist />
      </div>
      <div className="mt-4">
        <h4 className="font-semibold">Alerts</h4>
        <ManageAlerts />
      </div>
    </div>
  );
}


function ManageWatchlist(): JSX.Element {
  const [entries, setEntries] = useState<{ id: number; symbol: string }[]>([]);
  const [symbol, setSymbol] = useState('');

  async function load() {
    try {
      const res = await fetch('/api/watchlist/watchlist');
      const json = await res.json();
      setEntries(json || []);
    } catch (e) {}
  }

  useEffect(() => { load(); }, []);

  async function add() {
    if (!symbol) return;
    try {
      await fetch(`/api/watchlist/watchlist?symbol=${encodeURIComponent(symbol)}`, { method: 'POST' });
      setSymbol('');
      await load();
    } catch (e) {}
  }

  async function del(id: number) {
    try {
      await fetch(`/api/watchlist/watchlist/${id}`, { method: 'DELETE' });
      await load();
    } catch (e) {}
  }

  return (
    <div>
      <div className="flex space-x-2">
        <input value={symbol} onChange={(e) => setSymbol(e.target.value)} placeholder="Symbol e.g. BTCUSDT" className="border px-2 py-1" />
        <button onClick={add} className="bg-blue-600 text-white px-3 py-1 rounded">Add</button>
      </div>
      <ul className="mt-2">
        {entries.map((e) => (
          <li key={e.id} className="flex items-center space-x-2">
            <span className="font-mono">{e.symbol}</span>
            <button onClick={() => del(e.id)} className="text-red-600">Remove</button>
          </li>
        ))}
      </ul>
    </div>
  );
}


function ManageAlerts(): JSX.Element {
  const [alerts, setAlerts] = useState<any[]>([]);
  const [symbol, setSymbol] = useState('BTCUSDT');
  const [condition, setCondition] = useState('price_above');
  const [threshold, setThreshold] = useState('30000');

  async function load() {
    try {
      const res = await fetch('/api/watchlist/alerts');
      const json = await res.json();
      setAlerts(json || []);
    } catch (e) {}
  }
  useEffect(() => { load(); }, []);

  const [toast, setToast] = useState<string | null>(null);
  useEffect(() => {
    let mounted = true;
    const apiBase = (import.meta as any).env?.VITE_API_BASE_URL || '';
    let alertsWsUrl: string;
    if (apiBase) {
      const wsBase = apiBase.replace(/^http/, 'ws').replace(/\/$/, '');
      alertsWsUrl = `${wsBase}/watchlist/ws/alerts`;
    } else {
      alertsWsUrl = `${location.protocol === 'https:' ? 'wss' : 'ws'}://${location.host}/api/watchlist/ws/alerts`;
    }
    const ws = new WebSocket(alertsWsUrl);
    ws.onmessage = (ev) => {
      if (!mounted) return;
      try {
        const msg = JSON.parse(ev.data);
        if (msg && msg.type === 'alert_event') {
          setToast(`${msg.symbol} ${msg.condition} triggered (value=${msg.value})`);
        }
      } catch (e) {}
    };
    return () => { mounted = false; try { ws.close(); } catch (e) {} };
  }, []);

  async function add() {
    try {
      await fetch(`/api/watchlist/alerts?symbol=${encodeURIComponent(symbol)}&condition=${encodeURIComponent(condition)}&threshold=${encodeURIComponent(threshold)}`, { method: 'POST' });
      await load();
    } catch (e) {}
  }

  async function del(id: number) {
    try {
      await fetch(`/api/watchlist/alerts/${id}`, { method: 'DELETE' });
      await load();
    } catch (e) {}
  }

  return (
    <div>
      {toast ? <Toast message={toast} type="info" onClose={() => setToast(null)} /> : null}
      <div className="flex space-x-2 items-center">
  <input aria-label="alert-symbol" placeholder="Symbol" value={symbol} onChange={(e) => setSymbol(e.target.value)} className="border px-2 py-1" />
  <select aria-label="alert-condition" value={condition} onChange={(e) => setCondition(e.target.value)} className="border px-2 py-1">
          <option value="price_above">Price above</option>
          <option value="price_below">Price below</option>
          <option value="change_pct">Change % (abs)</option>
        </select>
  <input aria-label="alert-threshold" placeholder="Threshold" value={threshold} onChange={(e) => setThreshold(e.target.value)} className="border px-2 py-1" />
        <button onClick={add} className="bg-green-600 text-white px-3 py-1 rounded">Create</button>
      </div>
      <ul className="mt-2">
        {alerts.map((a) => (
          <li key={a.id} className="flex items-center space-x-2">
            <span>{a.symbol} {a.condition} {a.threshold}</span>
            <button onClick={() => del(a.id)} className="text-red-600">Remove</button>
          </li>
        ))}
      </ul>
    </div>
  );
}
