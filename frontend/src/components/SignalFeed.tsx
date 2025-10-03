import { useEffect, useMemo, useState } from "react";
import SignalDetail from "./SignalDetail";
import { fetchSignals, type Signal } from "../api/signals";

export type { Signal } from "../api/signals";

type Profile = "mixed" | "left" | "right";

type Props = {
  symbol?: string;
};

function toLocalString(iso: string): string {
  try {
    const d = new Date(iso);
    const time = d.toLocaleTimeString();
    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC";
    return `${time} ${tz}`;
  } catch (err) {
    return iso;
  }
}

export default function SignalFeed({ symbol: initialSymbol = "BTCUSDT" }: Props) {
  const [signals, setSignals] = useState<Signal[]>([]);
  const [selected, setSelected] = useState<Signal | null>(null);
  const [pageSize, setPageSize] = useState<number>(10);
  const [symbol, setSymbol] = useState<string>(initialSymbol);
  const [profile, setProfile] = useState<Profile>("mixed");
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [source, setSource] = useState<string | null>(null);
  const [lastUpdated, setLastUpdated] = useState<string | null>(null);

  useEffect(() => {
    let mounted = true;

    async function load() {
      try {
        setLoading(true);
        const { items, source } = await fetchSignals({ limit: pageSize, symbol, profile });
        if (!mounted) return;
        setSignals(items);
        setSource(source ?? null);
        setError(null);
        setLastUpdated(new Date().toLocaleTimeString());
      } catch (err) {
        if (!mounted) return;
        console.warn("signal fetch failed", err);
        setError("Failed to refresh signals from backend");
      } finally {
        if (mounted) setLoading(false);
      }
    }

    load();
    const id = window.setInterval(load, 5000);
    return () => {
      mounted = false;
      window.clearInterval(id);
    };
  }, [pageSize, symbol, profile]);

  const hint = useMemo(() => {
    if (!source) return null;
    if (source === "demo") {
      return "Demo signals (live market data disabled or unavailable)";
    }
    return "Live signals";
  }, [source]);


  useEffect(() => {
    setSelected(null);
  }, [symbol, profile]);
  return (
    <div className="p-2 border rounded space-y-2">
      <div className="flex items-center justify-between">
        <h3 className="font-semibold">Signal Feed</h3>
        {lastUpdated && (
          <span className="text-xs text-slate-500">Last updated: {lastUpdated}</span>
        )}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-2 text-sm">
        <label className="flex flex-col">
          Symbol
          <input
            value={symbol}
            onChange={(e) => setSymbol(e.target.value.toUpperCase())}
            className="text-sm p-1 border rounded"
            placeholder="BTCUSDT"
          />
        </label>
        <label className="flex flex-col">
          Profile
          <select
            value={profile}
            onChange={(e) => setProfile(e.target.value as Profile)}
            className="text-sm p-1 border rounded"
          >
            <option value="mixed">Mixed</option>
            <option value="left">Left (sell bias)</option>
            <option value="right">Right (buy bias)</option>
          </select>
        </label>
        <label className="flex flex-col">
          Page size
          <select
            value={pageSize}
            onChange={(e) => setPageSize(Number(e.target.value))}
            className="text-sm p-1 border rounded"
          >
            <option value={5}>5</option>
            <option value={10}>10</option>
            <option value={20}>20</option>
          </select>
        </label>
      </div>

      {hint && (
        <div className="text-xs text-slate-500" role="status">
          {hint}
        </div>
      )}

      {error && (
        <div className="text-xs text-amber-600" role="alert">
          {error}
        </div>
      )}

      {loading && !signals.length && (
        <div className="text-sm text-slate-500">Loading signals...</div>
      )}

      <ul className="space-y-2">
        {signals.map((s) => (
          <li
            key={s.id}
            className="p-2 border rounded hover:bg-slate-50 cursor-pointer"
            onClick={() => setSelected(s)}
          >
            <div className="flex justify-between items-start gap-3">
              <div>
                <div className="text-xs text-slate-500">{toLocalString(s.timestamp)}</div>
                <div className="flex items-baseline gap-3 mt-1">
                  <div className="font-mono text-sm">{s.symbol}</div>
                  <span
                    className={`px-2 py-0.5 text-xs rounded-full font-medium ${
                      s.direction === 'LONG'
                        ? 'bg-emerald-100 text-emerald-800'
                        : s.direction === 'SHORT'
                        ? 'bg-rose-100 text-rose-800'
                        : 'bg-slate-200 text-slate-700'
                    }`}
                  >
                    {s.direction}
                  </span>
                  <div className="text-sm">score: {Number.isFinite(s.score) ? s.score.toFixed(2) : '-'}
                  </div>
                  {s.confidence !== undefined && (
                    <div className="text-xs text-slate-500">{Math.round(s.confidence * 100)}%</div>
                  )}
                </div>
                {s.details?.note && (
                  <div className="text-xs text-slate-400 mt-1">{String(s.details.note)}</div>
                )}
                {/* Visual score/confidence bars */}
                <div className="mt-2 space-y-1">
                  <div className="flex items-center justify-between text-xs text-slate-500">
                    <span>Score</span>
                    <span>{Number.isFinite(s.score) ? Math.round(s.score * 100) + '%' : '-'}</span>
                  </div>
                  <div className="w-full bg-slate-100 h-2 rounded overflow-hidden">
                    <div
                      className={`h-2 ${s.direction === 'LONG' ? 'bg-emerald-500' : s.direction === 'SHORT' ? 'bg-rose-500' : 'bg-slate-400'}`}
                      style={{ width: `${Math.min(100, Math.max(0, (Number.isFinite(s.score) ? s.score * 100 : 0)))}%` }}
                    />
                  </div>
                  {typeof s.confidence === 'number' && (
                    <>
                      <div className="flex items-center justify-between text-xs text-slate-500 mt-1">
                        <span>Confidence</span>
                        <span>{Math.round(s.confidence * 100)}%</span>
                      </div>
                      <div className="w-full bg-slate-100 h-2 rounded overflow-hidden">
                        <div className="h-2 bg-indigo-500" style={{ width: `${Math.round(s.confidence * 100)}%` }} />
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          </li>
        ))}
      </ul>

      <SignalDetail signal={selected} onClose={() => setSelected(null)} />
      {!loading && !signals.length && (
        <div className="text-sm text-slate-500">No signals yet</div>
      )}
    </div>
  );
}
