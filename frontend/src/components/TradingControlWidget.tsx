import { useState, useEffect } from 'react';
import { Play, Square, BarChart3, DollarSign, TrendingUp, Settings } from 'lucide-react';

interface TradingStatus {
  is_running: boolean;
  balances: Record<string, number>;
  recent_trades: Array<{
    symbol: string;
    side: string;
    qty: number;
    price: number;
    timestamp: string;
  }>;
  ai_model_loaded: boolean;
  trading_symbols_count: number;
}

interface SystemEnvStatus {
  has_binance_keys?: boolean;
  binance_testnet?: boolean;
  real_trading_enabled?: boolean;
}

interface TradingConfig {
  max_position_size_usdc: number;
  min_confidence_threshold: number;
  risk_per_trade: number;
}

export default function TradingControlWidget() {
  const [status, setStatus] = useState<TradingStatus | null>(null);
  const [config, setConfig] = useState<TradingConfig>({
    max_position_size_usdc: 100,
    min_confidence_threshold: 0.65,
    risk_per_trade: 0.02
  });
  const [envStatus, setEnvStatus] = useState<SystemEnvStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [actionLoading, setActionLoading] = useState<string | null>(null);
  const [showConfig, setShowConfig] = useState(false);

  const fetchStatus = async () => {
    try {
      // In development, use relative path so Vite proxy works
      const isDev = (import.meta as any).env?.DEV;
      const apiBase = isDev ? '' : ((import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000');
      const [tradeResp, envResp] = await Promise.all([
        fetch(`${apiBase}/api/v1/trading/status`),
        fetch(`${apiBase}/api/v1/system/status`)
      ]);
      if (tradeResp.ok) {
        setStatus(await tradeResp.json());
      }
      if (envResp.ok) {
        setEnvStatus(await envResp.json());
      }
    } catch (error) {
      console.error('Failed to fetch trading status:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Update every 5 seconds
    return () => clearInterval(interval);
  }, []);

  const startTrading = async () => {
    setActionLoading('start');
    try {
      const isDev = (import.meta as any).env?.DEV;
      const apiBase = isDev ? '' : ((import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000');
      const response = await fetch(`${apiBase}/api/v1/trading/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ interval_minutes: 5 })
      });
      if (response.ok) {
        await fetchStatus();
      }
    } catch (error) {
      console.error('Failed to start trading:', error);
    } finally {
      setActionLoading(null);
    }
  };

  const stopTrading = async () => {
    setActionLoading('stop');
    try {
      const isDev = (import.meta as any).env?.DEV;
      const apiBase = isDev ? '' : ((import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000');
      const response = await fetch(`${apiBase}/api/v1/trading/stop`, { method: 'POST' });
      if (response.ok) {
        await fetchStatus();
      }
    } catch (error) {
      console.error('Failed to stop trading:', error);
    } finally {
      setActionLoading(null);
    }
  };

  const runCycle = async () => {
    setActionLoading('cycle');
    try {
      const isDev = (import.meta as any).env?.DEV;
      const apiBase = isDev ? '' : ((import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000');
      const response = await fetch(`${apiBase}/api/v1/trading/run-cycle`, { method: 'POST' });
      if (response.ok) {
        await fetchStatus();
      }
    } catch (error) {
      console.error('Failed to run trading cycle:', error);
    } finally {
      setActionLoading(null);
    }
  };

  const updateConfig = async () => {
    try {
      const isDev = (import.meta as any).env?.DEV;
      const apiBase = isDev ? '' : ((import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000');
      const response = await fetch(`${apiBase}/api/v1/trading/update-config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(config)
      });
      if (response.ok) {
        setShowConfig(false);
      }
    } catch (error) {
      console.error('Failed to update config:', error);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="flex items-center justify-center h-full text-red-500">
        <span className="text-sm">Failed to load trading status</span>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <BarChart3 className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          <span className="font-semibold text-gray-900 dark:text-white">AI Trading Engine</span>
          {envStatus && (
            <EnvBadge env={envStatus} />
          )}
        </div>
        <div className={`flex items-center space-x-1 text-xs px-2 py-1 rounded ${
          status.is_running 
            ? 'bg-green-100 text-green-700 dark:bg-green-900/20 dark:text-green-400'
            : 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-400'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            status.is_running ? 'bg-green-500' : 'bg-gray-400'
          }`}></div>
          <span>{status.is_running ? 'ACTIVE' : 'STOPPED'}</span>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center space-x-2">
        <button
          onClick={status.is_running ? stopTrading : startTrading}
          disabled={actionLoading === 'start' || actionLoading === 'stop'}
          className={`flex items-center space-x-1 px-3 py-1 rounded text-sm font-medium ${
            status.is_running
              ? 'bg-red-600 text-white hover:bg-red-700'
              : 'bg-green-600 text-white hover:bg-green-700'
          } disabled:opacity-50`}
        >
          {status.is_running ? <Square className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          <span>{status.is_running ? 'Stop' : 'Start'} Trading</span>
        </button>

        <button
          onClick={runCycle}
          disabled={actionLoading === 'cycle'}
          className="flex items-center space-x-1 px-3 py-1 rounded text-sm bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50"
        >
          <TrendingUp className="w-4 h-4" />
          <span>Run Cycle</span>
        </button>

        <button
          onClick={() => setShowConfig(!showConfig)}
          className="flex items-center space-x-1 px-3 py-1 rounded text-sm bg-gray-600 text-white hover:bg-gray-700"
        >
          <Settings className="w-4 h-4" />
          <span>Config</span>
        </button>
      </div>

      {/* Config Panel */}
      {showConfig && (
        <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3 space-y-3">
          <h4 className="text-sm font-semibold text-gray-900 dark:text-white">Trading Configuration</h4>
          
          <div className="grid grid-cols-1 gap-3 text-xs">
            <label className="flex flex-col space-y-1">
              <span className="text-gray-600 dark:text-gray-400">Max Position Size (USDC)</span>
              <input
                type="number"
                value={config.max_position_size_usdc}
                onChange={e => setConfig({...config, max_position_size_usdc: Number(e.target.value)})}
                className="px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            </label>
            
            <label className="flex flex-col space-y-1">
              <span className="text-gray-600 dark:text-gray-400">Min Confidence Threshold</span>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={config.min_confidence_threshold}
                onChange={e => setConfig({...config, min_confidence_threshold: Number(e.target.value)})}
                className="px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            </label>
            
            <label className="flex flex-col space-y-1">
              <span className="text-gray-600 dark:text-gray-400">Risk Per Trade (%)</span>
              <input
                type="number"
                step="0.001"
                min="0"
                max="0.1"
                value={config.risk_per_trade}
                onChange={e => setConfig({...config, risk_per_trade: Number(e.target.value)})}
                className="px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
              />
            </label>
          </div>
          
          <div className="flex space-x-2">
            <button
              onClick={updateConfig}
              className="px-3 py-1 bg-blue-600 text-white rounded text-xs hover:bg-blue-700"
            >
              Save Config
            </button>
            <button
              onClick={() => setShowConfig(false)}
              className="px-3 py-1 bg-gray-600 text-white rounded text-xs hover:bg-gray-700"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Status Grid */}
      <div className="grid grid-cols-2 gap-3">
        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">AI Model</div>
          <div className={`text-sm font-semibold ${
            status.ai_model_loaded ? 'text-green-600' : 'text-red-600'
          }`}>
            {status.ai_model_loaded ? 'Loaded' : 'Not Loaded'}
          </div>
        </div>

        <div className="bg-white dark:bg-gray-700 rounded-lg p-3 border border-gray-200 dark:border-gray-600">
          <div className="text-xs text-gray-500 dark:text-gray-400 mb-1">Trading Pairs</div>
          <div className="text-sm font-semibold text-blue-600">
            {status.trading_symbols_count}
          </div>
        </div>
      </div>

      {/* Balances */}
      <div className="bg-gradient-to-r from-green-50 to-blue-50 dark:from-green-900/20 dark:to-blue-900/20 rounded-lg p-3">
        <div className="flex items-center space-x-2 mb-2">
          <DollarSign className="w-4 h-4 text-green-600 dark:text-green-400" />
          <span className="text-sm font-semibold text-gray-900 dark:text-white">Account Balances</span>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          {Object.entries(status.balances || {}).map(([key, value]) => (
            <div key={key} className="flex justify-between">
              <span className="text-gray-600 dark:text-gray-400">{key}:</span>
              <span className="font-semibold text-gray-900 dark:text-white">{value.toFixed(2)}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Trades */}
      {status.recent_trades && status.recent_trades.length > 0 && (
        <div className="flex-1 overflow-hidden">
          <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">Recent AI Trades</h4>
          <div className="space-y-1 overflow-y-auto max-h-32">
            {status.recent_trades.slice(0, 5).map((trade, index) => (
              <div 
                key={index}
                className="flex items-center justify-between text-xs bg-gray-50 dark:bg-gray-800 rounded px-2 py-1"
              >
                <div className="flex items-center space-x-2">
                  <span className="font-mono text-gray-700 dark:text-gray-300">{trade.symbol}</span>
                  <span className={`px-1 rounded text-xs ${
                    trade.side === 'BUY' 
                      ? 'bg-green-100 text-green-800 dark:bg-green-900/20 dark:text-green-400' 
                      : 'bg-red-100 text-red-800 dark:bg-red-900/20 dark:text-red-400'
                  }`}>
                    {trade.side}
                  </span>
                </div>
                <div className="text-right">
                  <div className="font-semibold">{trade.qty.toFixed(4)}</div>
                  <div className="text-gray-500">${trade.price.toFixed(2)}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function EnvBadge({ env }: { env: SystemEnvStatus }) {
  let label = 'NO KEYS';
  let cls = 'bg-gray-200 text-gray-700 dark:bg-gray-700 dark:text-gray-300';
  if (env.has_binance_keys) {
    if (env.real_trading_enabled) {
      if (env.binance_testnet) {
        label = 'TESTNET LIVE';
        cls = 'bg-purple-600 text-white';
      } else {
        label = 'LIVE';
        cls = 'bg-green-600 text-white';
      }
    } else {
      if (env.binance_testnet) {
        label = 'TESTNET DRY-RUN';
        cls = 'bg-amber-500 text-white';
      } else {
        label = 'DRY-RUN';
        cls = 'bg-blue-500 text-white';
      }
    }
  }
  return (
    <span className={`text-[10px] px-2 py-0.5 rounded font-semibold tracking-wide ${cls}`}>{label}</span>
  );
}