import { useState, useEffect, useRef } from 'react';
import { useWebSocket } from './hooks/useWebSocket';
import TradingControls from './components/TradingControls';
import AITradingControls from './components/AITradingControls';
import AISignals from './components/AISignals';
import ContinuousLearningDashboard from './components/ContinuousLearningDashboard';
import EnhancedDataDashboard from './components/EnhancedDataDashboard';
import SimpleChart from './components/SimpleChart';
import ChatPanel from './components/ChatPanel';
import CoinTable from './components/CoinTable';

interface DashboardData {
  systemStatus: any;
  marketData: any;
  portfolio: any;
  signals: any[];
  trades: any[];
  chartData: any[];
  trading?: any;
  stats?: any;
  marketTicks?: any[];
  orderbook?: any;
  stream_meta?: any;
}

export default function SimpleDashboard() {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true); // bare brukt for første last
  const [initialLoad, setInitialLoad] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [tradingStatus, setTradingStatus] = useState<'active' | 'inactive' | 'unknown'>('unknown');
  const [aiTradingStatus, setAITradingStatus] = useState<'active' | 'inactive' | 'unknown'>('unknown');
  const [actionLoading, setActionLoading] = useState(false);

  // WebSocket for real-time updates
  const { data: wsData, connectionStatus } = useWebSocket({
    url: 'ws://127.0.0.1:8000/ws/dashboard',
    enabled: true,
    debounceMs: 400
  });
  const lastWsUpdateRef = useRef<number | null>(null);
  const [staleWs, setStaleWs] = useState(false);

  // Track WS staleness
  useEffect(() => {
    if (wsData) {
      lastWsUpdateRef.current = Date.now();
      if (staleWs) setStaleWs(false);
    }
  }, [wsData]);

  useEffect(() => {
    const id = setInterval(() => {
      if (lastWsUpdateRef.current) {
        const diff = Date.now() - lastWsUpdateRef.current;
        if (diff > 5000 && connectionStatus === 'connected') {
          setStaleWs(true);
        }
      }
    }, 2000);
    return () => clearInterval(id);
  }, [connectionStatus]);

  // Trading functions
  const startTrading = async () => {
    setActionLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/trading/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ interval_minutes: 5 })
      });
      if (response.ok) {
        setTradingStatus('active');
      }
    } catch (err) {
      console.error('Failed to start trading:', err);
    } finally {
      setActionLoading(false);
    }
  };

  const stopTrading = async () => {
    setActionLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/trading/stop', {
        method: 'POST'
      });
      if (response.ok) {
        setTradingStatus('inactive');
      }
    } catch (err) {
      console.error('Failed to stop trading:', err);
    } finally {
      setActionLoading(false);
    }
  };

  const runCycle = async () => {
    setActionLoading(true);
    try {
      await fetch('http://127.0.0.1:8000/api/v1/trading/run-cycle', {
        method: 'POST'
      });
    } catch (err) {
      console.error('Failed to run cycle:', err);
    } finally {
      setActionLoading(false);
    }
  };

  // AI Trading functions
  const startAITrading = async (symbols: string[]) => {
    setActionLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/ai-trading/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ symbols })
      });
      if (response.ok) {
        setAITradingStatus('active');
      }
    } catch (err) {
      console.error('Failed to start AI trading:', err);
    } finally {
      setActionLoading(false);
    }
  };

  const stopAITrading = async () => {
    setActionLoading(true);
    try {
      const response = await fetch('http://127.0.0.1:8000/api/v1/ai-trading/stop', {
        method: 'POST'
      });
      if (response.ok) {
        setAITradingStatus('inactive');
      }
    } catch (err) {
      console.error('Failed to stop AI trading:', err);
    } finally {
      setActionLoading(false);
    }
  };

  const updateAIConfig = async (config: any) => {
    setActionLoading(true);
    try {
      await fetch('http://127.0.0.1:8000/api/v1/ai-trading/config', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
    } catch (err) {
      console.error('Failed to update AI config:', err);
    } finally {
      setActionLoading(false);
    }
  };

  // Fetch AI trading status on component mount
  useEffect(() => {
    const fetchAIStatus = async () => {
      try {
        const response = await fetch('http://127.0.0.1:8000/api/v1/ai-trading/status');
        if (response.ok) {
          const data = await response.json();
          setAITradingStatus(data.enabled ? 'active' : 'inactive');
        }
      } catch (err) {
        console.error('Failed to fetch AI trading status:', err);
      }
    };
    
    fetchAIStatus();
  }, []);

  // WebSocket: inkrementell merge uten å trigge unødig re-render
  useEffect(() => {
    if (!wsData) return;
    setData(prev => {
      const base = prev || {
        systemStatus: null,
        marketData: null,
        portfolio: null,
        signals: [],
        trades: [],
        chartData: [],
        trading: null
      };
      // Shallow differ for å unngå ny referanse hvis identisk
      const next = { ...base } as any;
  if (wsData.systemStatus !== undefined) next.systemStatus = wsData.systemStatus;
  if (wsData.marketData !== undefined) next.marketData = wsData.marketData;
  if (wsData.portfolio !== undefined) next.portfolio = wsData.portfolio;
  if (wsData.signals !== undefined) next.signals = wsData.signals;
  if (wsData.trades !== undefined) next.trades = wsData.trades;
  if (wsData.chartData !== undefined) next.chartData = wsData.chartData;
  if (wsData.stats !== undefined) next.stats = wsData.stats;
      if (wsData.trading) {
        // Normaliser balances dersom dict
        if (wsData.trading.balances && !Array.isArray(wsData.trading.balances) && typeof wsData.trading.balances === 'object') {
          const arr = Object.entries(wsData.trading.balances).map(([k,v]: any) => ({ asset: k, balance: v }));
          wsData.trading.balances = arr;
        }
        next.trading = wsData.trading;
        if (typeof wsData.trading.is_running === 'boolean') {
          setTradingStatus(wsData.trading.is_running ? 'active' : 'inactive');
        }
      }
      if ('marketTicks' in wsData) next.marketTicks = wsData.marketTicks;
      return next;
    });
  }, [wsData]);

  // Polling fallback (kun hvis WebSocket ikke er connected)
  useEffect(() => {
    let cancelled = false;
    async function fetchOnce(isInitial = false) {
      try {
        if (isInitial) {
            setLoading(true);
            setError(null);
        }
        const baseUrls = ['http://127.0.0.1:8000', ''];
        const results: any = {};
        for (const baseUrl of baseUrls) {
          try {
            const [systemRes, marketRes, portfolioRes, signalsRes] = await Promise.all([
              fetch(`${baseUrl}/api/v1/system/status`).catch(()=>null),
              fetch(`${baseUrl}/api/v1/portfolio/market-overview`).catch(()=>null),
              fetch(`${baseUrl}/api/v1/portfolio`).catch(()=>null),
              fetch(`${baseUrl}/api/v1/signals/recent?limit=5`).catch(()=>null)
            ]);
            if (systemRes?.ok) results.systemStatus = await systemRes.json();
            if (marketRes?.ok) results.marketData = await marketRes.json();
            if (portfolioRes?.ok) results.portfolio = await portfolioRes.json();
            if (signalsRes?.ok) results.signals = await signalsRes.json();
            if (Object.keys(results).length) break;
          } catch { /* prøv neste */ }
        }
        if (cancelled) return;
        setData(prev => ({
          systemStatus: results.systemStatus || prev?.systemStatus || { service: 'N/A', uptime_seconds: 0 },
          marketData: results.marketData || prev?.marketData || { marketCap: 0, volume24h: 0 },
            portfolio: results.portfolio || prev?.portfolio || { totalValue: 0, positions: [] },
            signals: results.signals || prev?.signals || [],
            trades: prev?.trades || [],
            chartData: prev?.chartData || []
        }));
      } catch (err: any) {
        if (!cancelled) setError(`Feil ved lasting av data: ${err.message}`);
      } finally {
        if (isInitial) {
          setLoading(false);
          setInitialLoad(false);
        }
      }
    }

    // Første kall alltid
    fetchOnce(true);
    if (connectionStatus === 'connected') {
      // Ikke poll når vi har stabil WebSocket
      return () => { cancelled = true; };
    }
    const id = setInterval(() => fetchOnce(false), 5000);
    return () => { cancelled = true; clearInterval(id); };
  }, [connectionStatus]);

  if (loading && initialLoad) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <p>Laster live data...</p>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-gray-900 text-white flex items-center justify-center">
        <div className="text-center">
          <h2 className="text-2xl font-bold text-red-500 mb-4">Connection Error</h2>
          <p className="text-gray-300">{error}</p>
          <button 
            onClick={() => window.location.reload()} 
            className="mt-4 px-6 py-2 bg-blue-600 rounded hover:bg-blue-700"
          >
            Prøv igjen
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white p-6">
      <div className="max-w-7xl mx-auto space-y-6">
        <header className="flex items-center justify-between bg-gray-800 rounded-lg p-4">
          <div>
            <h1 className="text-2xl font-bold">Quantum Trader Dashboard</h1>
            <div className="flex items-center space-x-3 text-sm mt-1">
              <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${data?.systemStatus?.binance_testnet ? 'bg-yellow-600 text-yellow-100' : 'bg-green-600 text-green-100'}`}>
                {data?.systemStatus?.binance_testnet ? 'TESTNET' : 'LIVE'}
              </span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${data?.trading?.is_running ? 'bg-emerald-600 text-emerald-100' : 'bg-gray-600 text-gray-200'}`}>
                {data?.trading?.is_running ? 'RUNNING' : 'IDLE'}
              </span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${data?.systemStatus?.has_binance_keys ? 'bg-blue-600 text-blue-100' : 'bg-red-600 text-red-100'}`}>
                Keys {data?.systemStatus?.has_binance_keys ? '✓' : '✗'}
              </span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${data?.trading?.ai_model_loaded ? 'bg-purple-600 text-purple-100' : 'bg-gray-600 text-gray-200'}`}>
                AI {data?.trading?.ai_model_loaded ? 'READY' : 'LOADING'}
              </span>
              {typeof data?.stats?.daily_pnl_change === 'number' && (
                <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${data?.stats?.daily_pnl_change >=0 ? 'bg-green-700 text-green-100' : 'bg-red-700 text-red-100'}`}>
                  24h PnL {data?.stats?.daily_pnl_change}
                </span>
              )}
              {typeof data?.stats?.pnl_percent === 'number' && (
                <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${data?.stats?.pnl_percent >=0 ? 'bg-green-800 text-green-100' : 'bg-red-800 text-red-100'}`}>
                  Total {data?.stats?.pnl_percent}%
                </span>
              )}
            </div>
          </div>
          <div className="flex items-center space-x-4 text-sm">
            <div className="flex items-center space-x-2">
              <div className={`w-3 h-3 rounded-full ${
                connectionStatus === 'connected' ? (staleWs ? 'bg-yellow-500' : 'bg-green-500') :
                connectionStatus === 'connecting' ? 'bg-yellow-500' : 'bg-red-500'
              }`}></div>
              <span>{connectionStatus === 'connected' && staleWs ? 'WebSocket stale - venter...' : `WebSocket: ${connectionStatus}`}</span>
            </div>
            <div>Last update: {lastWsUpdateRef.current ? new Date(lastWsUpdateRef.current).toLocaleTimeString() : '-'}</div>
          </div>
        </header>

        <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-green-400">System Status</h2>
            <div className="space-y-2">
              <p><span className="text-gray-400">Service:</span> {data?.systemStatus?.service}</p>
              <p><span className="text-gray-400">Uptime:</span> {Math.floor((data?.systemStatus?.uptime_seconds || 0) / 60)} min</p>
              <p><span className="text-gray-400">Binance Keys:</span> {data?.systemStatus?.has_binance_keys ? '✓' : '✗'}</p>
              <p><span className="text-gray-400">Testnet:</span> {data?.systemStatus?.binance_testnet ? 'Yes' : 'No'}</p>
              <p><span className="text-gray-400">Status:</span> <span className="text-green-400">●</span> Online</p>
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-blue-400">Market Overview</h2>
            <div className="space-y-2">
              <p><span className="text-gray-400">Market Cap:</span> ${(data?.marketData?.marketCap || 0).toLocaleString()}</p>
              <p><span className="text-gray-400">24h Volume:</span> ${(data?.marketData?.volume24h || 0).toLocaleString()}</p>
              <p><span className="text-gray-400">Fear & Greed:</span> {data?.marketData?.fearGreedIndex || 'N/A'}</p>
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-yellow-400">Portfolio</h2>
            <div className="space-y-2">
              <p><span className="text-gray-400">Total Value:</span> ${(data?.portfolio?.totalValue || 0).toLocaleString()}</p>
              <p><span className="text-gray-400">Positions:</span> {data?.portfolio?.positions?.length || 0}</p>
              <p><span className="text-gray-400">P&L:</span> {data?.portfolio?.totalPnL || 0}%</p>
            </div>
          </div>
          <TradingControls
            onStartTrading={startTrading}
            onStopTrading={stopTrading}
            onRunCycle={runCycle}
            tradingStatus={tradingStatus}
            loading={actionLoading}
          />
        </section>

        {/* AI Trading Section */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AITradingControls
            onStartAITrading={startAITrading}
            onStopAITrading={stopAITrading}
            onUpdateAIConfig={updateAIConfig}
            aiTradingStatus={aiTradingStatus}
            loading={actionLoading}
          />
          <AISignals />
        </section>

        {/* Continuous Learning Engine Section */}
        <section className="w-full">
          <ContinuousLearningDashboard />
        </section>

        {/* Enhanced Multi-Source Data Section */}
        <section className="w-full">
          <EnhancedDataDashboard wsUrl="ws://127.0.0.1:8000/ws/enhanced-data" />
        </section>

        {/* Advanced Metrics Row */}
        <section className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-3 text-teal-400">Trading Engine</h2>
            <div className="space-y-2 text-sm">
              <p><span className="text-gray-400">Running:</span> {data?.trading?.is_running ? 'Yes' : 'No'}</p>
              <p><span className="text-gray-400">Symbols:</span> {data?.trading?.trading_symbols_count ?? '—'}</p>
              <p><span className="text-gray-400">AI Model:</span> {data?.trading?.ai_model_loaded ? 'Loaded' : 'Not Loaded'}</p>
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-3 text-pink-400">Risk & PnL</h2>
            <div className="space-y-2 text-sm">
              <p><span className="text-gray-400">Total Trades:</span> {data?.stats?.total_trades ?? '—'}</p>
              <p><span className="text-gray-400">Active Symbols:</span> {data?.stats?.active_symbols ?? '—'}</p>
              <p><span className="text-gray-400">Avg Price:</span> {data?.stats?.avg_price ?? '—'}</p>
              <p><span className="text-gray-400">PnL%:</span> {data?.stats?.pnl_percent ?? '—'}%</p>
              <p><span className="text-gray-400">24h PnL%:</span> {data?.stats?.daily_pnl_change_percent ?? '—'}%</p>
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-3 text-indigo-400">Account Balances</h2>
            <div className="space-y-1 text-xs max-h-40 overflow-y-auto">
              {Array.isArray(data?.trading?.balances) ? data?.trading?.balances.map((b:any,i:number)=>(
                <div key={i} className="flex justify-between border-b border-gray-700 py-1">
                  <span>{b.asset}</span>
                  <span className="text-gray-300">{b.free ?? b.balance}</span>
                </div>
              )) : <p className="text-gray-500">Ingen balanse-data</p>}
            </div>
          </div>
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-3 text-yellow-300">PnL / Symbol</h2>
            <div className="space-y-1 text-xs max-h-40 overflow-y-auto">
              {data?.stats?.pnl_per_symbol ? Object.entries(data.stats.pnl_per_symbol).sort((a:any,b:any)=> (b[1] as number)-(a[1] as number)).slice(0,10).map(([sym,val]:any)=>(
                <div key={sym} className="flex justify-between border-b border-gray-700 py-1">
                  <span>{sym}</span>
                  <span className={val>=0? 'text-green-400':'text-red-400'}>{val.toFixed ? val.toFixed(2) : val}</span>
                </div>
              )) : <p className="text-gray-500">Ingen PnL-data</p>}
            </div>
          </div>
        </section>

        <section className="grid grid-cols-1 xl:grid-cols-4 gap-6">
          <div className="xl:col-span-2 bg-gray-800 rounded-lg p-6">
            <SimpleChart
              data={data?.chartData || []}
              title="Price Chart"
              color="#3b82f6"
              height={300}
            />
          </div>
          <div className="bg-gray-800 rounded-lg p-6 order-last xl:order-none">
            <h2 className="text-xl font-semibold mb-4 text-purple-400">Recent Trades</h2>
            {(data?.trades?.length || 0) > 0 ? (
              <div className="space-y-2 max-h-64 overflow-y-auto">
                {data?.trades?.slice(0, 10).map((trade: any, i: number) => (
                  <div key={i} className="flex justify-between items-center p-2 bg-gray-700 rounded">
                    <span className="font-medium">{trade.symbol || 'N/A'}</span>
                    <span className={trade.side === 'BUY' ? 'text-green-400' : 'text-red-400'}>{trade.side || 'N/A'}</span>
                    <span>${trade.price || 0}</span>
                    <span className="text-sm text-gray-400">{trade.timestamp ? new Date(trade.timestamp).toLocaleTimeString() : 'N/A'}</span>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-gray-400">Ingen trades tilgjengelig</p>
            )}
          </div>
          <div className="bg-gray-800 rounded-lg p-6 space-y-4">
            <div className="flex items-center justify-between">
              <h2 className="text-xl font-semibold text-emerald-400">Live Ticks</h2>
              {data?.stats?.latency_high && (
                <span className="text-xs px-2 py-0.5 rounded-full bg-red-700 text-red-100">Latency High</span>
              )}
            </div>
            {(data?.marketTicks?.length || 0) > 0 ? (
              <div className="space-y-1 max-h-64 overflow-y-auto text-xs">
                {data?.marketTicks?.map((t:any)=> {
                  const srcColor = t.src === 'stream' ? 'bg-green-600' : t.src === 'rest' ? 'bg-blue-600' : 'bg-gray-600';
                  return (
                    <div key={t.symbol} className="flex items-center justify-between border-b border-gray-700 py-1">
                      <div className="flex items-center space-x-2">
                        <span className={`w-2 h-2 rounded-full ${srcColor}`}></span>
                        <span>{t.symbol}</span>
                      </div>
                      <div className="flex items-center space-x-3">
                        <span className={t.price ? 'text-gray-200':'text-gray-500'}>{t.price ? (t.price.toFixed ? t.price.toFixed(4): t.price) : '—'}</span>
                        {typeof t.age_ms === 'number' && <span className="text-[10px] text-gray-500">{Math.round(t.age_ms/1000)}s</span>}
                      </div>
                    </div>
                  );
                })}
              </div>
            ) : <p className="text-gray-500 text-sm">Ingen ticks</p>}
            <div className="text-[10px] text-gray-500 flex flex-wrap gap-2 pt-1">
              <span><span className="inline-block w-2 h-2 rounded-full bg-green-600 mr-1"></span>stream</span>
              <span><span className="inline-block w-2 h-2 rounded-full bg-blue-600 mr-1"></span>rest</span>
              <span><span className="inline-block w-2 h-2 rounded-full bg-gray-600 mr-1"></span>fallback</span>
            </div>
          </div>
        </section>

        {/* Coin Table + Signals Row */}
        <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <CoinTable />
          </div>
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 text-purple-400">Recent Signals</h2>
            {(data?.signals?.length || 0) > 0 ? (
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="text-gray-400 border-b border-gray-700">
                      <th className="text-left p-2">Symbol</th>
                      <th className="text-left p-2">Side</th>
                      <th className="text-left p-2">Confidence</th>
                      <th className="text-left p-2">Time</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data?.signals?.slice(0, 15).map((signal: any, i: number) => (
                      <tr key={i} className="border-b border-gray-700">
                        <td className="p-2 font-medium">{signal.symbol}</td>
                        <td className="p-2">
                          <span className={signal.side === 'BUY' ? 'text-green-400' : 'text-red-400'}>{signal.side}</span>
                        </td>
                        <td className="p-2">{signal.confidence}%</td>
                        <td className="p-2">{new Date(signal.ts || Date.now()).toLocaleTimeString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-gray-400">Ingen signaler tilgjengelig</p>
            )}
          </div>
        </section>

        <section className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="md:col-span-2 bg-gray-800 rounded-lg p-6">
            <h2 className="text-lg font-semibold mb-3 text-cyan-400">Orderbook (synthetic)</h2>
            {data?.orderbook?.bids ? (
              <div className="grid grid-cols-2 gap-4 text-xs">
                <div>
                  <h3 className="text-green-400 font-semibold mb-1">Bids</h3>
                  <div className="space-y-1 max-h-48 overflow-y-auto pr-2">
                    {data.orderbook.bids.map((l:any,i:number)=>(
                      <div key={i} className="flex justify-between border-b border-gray-700 py-0.5">
                        <span className="text-gray-300">{l[0]}</span>
                        <span className="text-green-400">{l[1]}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <h3 className="text-red-400 font-semibold mb-1">Asks</h3>
                  <div className="space-y-1 max-h-48 overflow-y-auto pr-2">
                    {data.orderbook.asks.map((l:any,i:number)=>(
                      <div key={i} className="flex justify-between border-b border-gray-700 py-0.5">
                        <span className="text-gray-300">{l[0]}</span>
                        <span className="text-red-400">{l[1]}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            ) : <p className="text-gray-500 text-sm">Ingen orderbookdata</p>}
            <div className="text-[10px] text-gray-500 mt-2">Primærsymbol: {data?.orderbook?.symbol}</div>
          </div>
            <ChatPanel />
        </section>

        <footer className="text-center text-gray-500 text-sm space-y-1">
          <div>WebSocket: {connectionStatus}{staleWs ? ' (stale)' : ''} • Sist WS: {lastWsUpdateRef.current ? new Date(lastWsUpdateRef.current).toLocaleTimeString() : '-'}</div>
          <div className="text-xs text-gray-600">Polling fallback aktiv kun når WS ikke er connected. Debounce 400ms.</div>
        </footer>
      </div>
    </div>
  );
}