import { useState, useEffect, useMemo, useRef } from 'react';

interface WatchlistEntry {
  symbol: string;
  price?: number;
  change24h?: number;
  volume24h?: number;
  sparkline?: number[];
  ts?: number | string;
  error?: string;
}

// Curated defaults (corrected SOLUSDT)
const DEFAULT_SYMBOLS = [
  'BTCUSDT','ETHUSDT','BNBUSDT','SOLUSDT','XRPUSDT','ADAUSDT','AVAXUSDT','MATICUSDT','DOTUSDT','LINKUSDT'
];

const PINNED_COUNT = 10; // first 10 are pinned
const STORAGE_KEY = 'coinTableSymbols_v2';
const STORAGE_VERSION = 2;
const AUTOCOMPLETE_DELAY = 600; // ms

export default function CoinTable() {
  const [symbols, setSymbols] = useState<string[]>(DEFAULT_SYMBOLS);
  const [query, setQuery] = useState('');
  const [collapsed, setCollapsed] = useState(false);
  const [sortCol, setSortCol] = useState<'symbol'|'price'|'change24h'|'volume24h'>('volume24h');
  const [sortDir, setSortDir] = useState<'asc'|'desc'>('desc');
  const debounceRef = useRef<number | null>(null);
  const [rows, setRows] = useState<WatchlistEntry[]>([]);

  // Fetch watchlist data from HTTP API instead of WebSocket
  useEffect(() => {
    const fetchWatchlistData = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/v1/watchlist');
        if (response.ok) {
          const data = await response.json();
          if (Array.isArray(data)) {
            setRows(data as WatchlistEntry[]);
          }
        }
      } catch (error) {
        console.error('Error fetching watchlist data:', error);
      }
    };

    // Initial fetch
    fetchWatchlistData();

    // Fetch every 5 seconds
    const interval = setInterval(fetchWatchlistData, 5000);

    return () => clearInterval(interval);
  }, [symbols]);

  // Load persisted symbols
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      if (stored) {
        const obj = JSON.parse(stored);
        if (obj && obj.version === STORAGE_VERSION && Array.isArray(obj.symbols)) {
          const pinned = DEFAULT_SYMBOLS;
          const rest = obj.symbols.filter((s:string)=> !pinned.includes(s));
          setSymbols([...pinned, ...rest]);
        } else {
          localStorage.removeItem(STORAGE_KEY); // purge old format
        }
      }
    } catch {/* ignore */}
  }, []);

  // Persist symbols (exclude duplicates automatically)
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ version: STORAGE_VERSION, symbols }));
    } catch {/* ignore */}
  }, [symbols]);

  const addSymbol = (candidate?: string) => {
    const raw = (candidate ?? query).toUpperCase().trim();
    if (!raw) return;
    if (!/^[A-Z0-9]{5,20}$/.test(raw)) return; // naive validation
    if (symbols.includes(raw)) return;
    setSymbols(prev => [...prev, raw]);
    if (!candidate) setQuery('');
  };

  // Search-as-you-type debounce auto-add
  useEffect(() => {
    if (!query) return;
    if (debounceRef.current) window.clearTimeout(debounceRef.current);
    debounceRef.current = window.setTimeout(()=> addSymbol(query), AUTOCOMPLETE_DELAY);
    return () => { if (debounceRef.current) window.clearTimeout(debounceRef.current); };
  }, [query]);

  const removeSymbol = (sym: string) => {
    const idx = symbols.indexOf(sym);
    if (idx > -1 && idx < PINNED_COUNT) return; // pinned
    setSymbols(prev => prev.filter(s => s !== sym));
  };

  const sorted = useMemo(() => {
    const arr = [...rows];
    arr.sort((a,b) => {
      if (sortCol === 'symbol') {
        return sortDir === 'asc' ? a.symbol.localeCompare(b.symbol) : b.symbol.localeCompare(a.symbol);
      }
      const vA = typeof a[sortCol] === 'number' ? a[sortCol] as number : 0;
      const vB = typeof b[sortCol] === 'number' ? b[sortCol] as number : 0;
      if (sortDir === 'asc') return vA - vB;
      return vB - vA;
    });
    return arr;
  }, [rows, sortCol, sortDir]);

  return (
    <div className={`bg-gray-800 rounded-lg p-6 h-full flex flex-col ${collapsed ? 'max-h-32 overflow-hidden' : ''}`}> 
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-xl font-semibold text-orange-400">Coins</h2>
        <div className="flex space-x-2 items-center">
          <input
            value={query}
            onChange={e=> setQuery(e.target.value.toUpperCase())}
            onKeyDown={e=> { if (e.key==='Enter') addSymbol(); }}
            placeholder="Type symbol (auto-add)"
            className="bg-gray-700 text-xs rounded px-2 py-1 focus:outline-none w-44 tracking-wide"
          />
          <button onClick={()=>addSymbol()} className="text-xs px-2 py-1 bg-orange-600 hover:bg-orange-700 rounded">Add</button>
          <button onClick={()=>setCollapsed(c=>!c)} className="text-xs px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded">{collapsed ? 'Expand' : 'Collapse'}</button>
        </div>
      </div>
      <div className="overflow-x-auto flex-1">
        <table className="w-full text-xs select-none">
          <thead>
            <tr className="text-gray-400 border-b border-gray-700">
              <th className="p-2 text-left">#</th>
              <th className="p-2 text-left cursor-pointer" onClick={()=>{
                setSortCol('symbol'); setSortDir(sortCol==='symbol'&&sortDir==='desc'?'asc':'desc');
              }}>Symbol {sortCol==='symbol'?(sortDir==='asc'?'▲':'▼'):''}</th>
              <th className="p-2 text-right cursor-pointer" onClick={()=>{
                setSortCol('price'); setSortDir(sortCol==='price'&&sortDir==='desc'?'asc':'desc');
              }}>Price {sortCol==='price'?(sortDir==='asc'?'▲':'▼'):''}</th>
              <th className="p-2 text-right cursor-pointer" onClick={()=>{
                setSortCol('change24h'); setSortDir(sortCol==='change24h'&&sortDir==='desc'?'asc':'desc');
              }}>24h % {sortCol==='change24h'?(sortDir==='asc'?'▲':'▼'):''}</th>
              <th className="p-2 text-right cursor-pointer" onClick={()=>{
                setSortCol('volume24h'); setSortDir(sortCol==='volume24h'&&sortDir==='desc'?'asc':'desc');
              }}>24h Vol {sortCol==='volume24h'?(sortDir==='asc'?'▲':'▼'):''}</th>
              <th className="p-2 text-left">Spark</th>
              <th className="p-2 text-center">&nbsp;</th>
            </tr>
          </thead>
          <tbody>
            {sorted.slice(0, 150).map((row, i) => {
              const pct = row.change24h != null ? (row.change24h * 100) : undefined;
              const pctCls = pct == null ? 'text-gray-400' : pct >= 0 ? 'text-green-400' : 'text-red-400';
              const spark = row.sparkline ? row.sparkline.slice(-16) : [];
              // SVG sparkline
              const svgW = 60, svgH = 18;
              let svgPath = '';
              if (spark.length > 1) {
                const min = Math.min(...spark), max = Math.max(...spark);
                const norm = (v:number) => max-min===0 ? svgH/2 : svgH-(svgH*(v-min)/(max-min));
                svgPath = spark.map((v,si)=>`${si===0?'M':'L'}${(si/(spark.length-1))*svgW},${norm(v)}`).join(' ');
              }
              return (
                <tr key={row.symbol} className="border-b border-gray-800 hover:bg-gray-700/30">
                  <td className="p-2 text-gray-500">{i+1}</td>
                  <td className="p-2 font-medium flex items-center space-x-1">
                    <span>{row.symbol}</span>
                    {symbols.indexOf(row.symbol) < PINNED_COUNT && <span className="text-[9px] text-yellow-500">★</span>}
                  </td>
                  <td className="p-2 text-right tabular-nums">{row.error ? '—' : (row.price != null ? row.price.toFixed(4) : '—')}</td>
                  <td className={`p-2 text-right tabular-nums ${pctCls}`}>{pct == null ? '—' : pct.toFixed(2)}</td>
                  <td className="p-2 text-right tabular-nums">{row.volume24h ? Math.round(row.volume24h).toLocaleString() : '—'}</td>
                  <td className="p-2 text-left text-[10px] text-gray-400">
                    {spark.length > 1 ? (
                      <svg width={svgW} height={svgH} viewBox={`0 0 ${svgW} ${svgH}`} className="inline-block align-middle bg-transparent">
                        <path d={svgPath} stroke="#f59e42" strokeWidth="2" fill="none" />
                      </svg>
                    ) : '—'}
                  </td>
                  <td className="p-2 text-center">
                    {symbols.indexOf(row.symbol) >= PINNED_COUNT && (
                      <button onClick={()=>removeSymbol(row.symbol)} className="text-red-500 hover:text-red-400 text-[10px]">✕</button>
                    )}
                  </td>
                </tr>
              );
            })}
            {sorted.length === 0 && (
              <tr><td colSpan={7} className="p-4 text-center text-gray-500">No data</td></tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="mt-2 text-[10px] text-gray-500">Showing {sorted.length} symbols • Pinned first {PINNED_COUNT}. Type to auto-add. Persisted locally. {collapsed ? 'Table collapsed.' : ''}</div>
    </div>
  );
}
