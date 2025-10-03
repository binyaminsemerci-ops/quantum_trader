import { useEffect, useState } from 'react';
import { Server, Activity, Timer, Brain, PlayCircle } from 'lucide-react';
import { fetchWithFallback } from '../utils/net';

interface SystemStatus {
  service: string;
  version?: string;
  uptime_seconds: number;
  timestamp: string;
  training?: any;
  trading?: any;
  evaluator_running?: boolean;
  market_data_cache_age_sec?: number | null;
}

function formatDuration(sec: number): string {
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

export default function StatusWidget() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const load = async () => {
    const res = await fetchWithFallback<SystemStatus>('/api/v1/system/status');
    if (res.ok && res.data) {
      setStatus(res.data);
      setError(null);
    } else {
      setError(res.error || 'Failed');
    }
    setLoading(false);
  };

  useEffect(() => {
    load();
    const id = setInterval(load, 5000);
    return () => clearInterval(id);
  }, []);

  if (loading) {
    return <div className="flex items-center justify-center h-full"><div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-500"/></div>;
  }
  if (!status) {
    return <div className="text-xs text-red-500 p-2">Status utilgjengelig{error?`: ${error}`:''}</div>;
  }

  const trainingEnabled = status.training?.enabled;
  const tradingEnabled = status.trading?.enabled;

  return (
    <div className="h-full flex flex-col space-y-3 text-xs">
      <div className="flex items-center space-x-2">
        <Server className="w-4 h-4 text-blue-600" />
        <span className="font-semibold text-gray-900 dark:text-white">System Status</span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        <div className="bg-white dark:bg-gray-700 rounded p-2 border border-gray-200 dark:border-gray-600">
          <div className="flex items-center space-x-1 mb-1"><Activity className="w-3 h-3 text-green-500"/><span>Uptime</span></div>
          <div className="font-mono">{formatDuration(status.uptime_seconds)}</div>
        </div>
        <div className="bg-white dark:bg-gray-700 rounded p-2 border border-gray-200 dark:border-gray-600">
          <div className="flex items-center space-x-1 mb-1"><Timer className="w-3 h-3 text-indigo-500"/><span>Cache</span></div>
          <div>{status.market_data_cache_age_sec!=null?`${Math.round(status.market_data_cache_age_sec)}s`:'-'}</div>
        </div>
        <div className="bg-white dark:bg-gray-700 rounded p-2 border border-gray-200 dark:border-gray-600 col-span-1">
          <div className="flex items-center space-x-1 mb-1"><Brain className="w-3 h-3 text-purple-500"/><span>Auto-Train</span></div>
          <div className={trainingEnabled? 'text-green-600':'text-gray-500'}>{trainingEnabled?`ON ${Math.round((status.training.interval_sec||0)/60)}m`:'OFF'}</div>
        </div>
        <div className="bg-white dark:bg-gray-700 rounded p-2 border border-gray-200 dark:border-gray-600 col-span-1">
          <div className="flex items-center space-x-1 mb-1"><PlayCircle className="w-3 h-3 text-amber-500"/><span>Auto-Trade</span></div>
          <div className={tradingEnabled? 'text-green-600':'text-gray-500'}>{tradingEnabled?`ON ${Math.round((status.trading.interval_sec||0)/60)}m`:'OFF'}</div>
        </div>
      </div>
      <div className="mt-auto text-[10px] text-gray-500 dark:text-gray-400">{status.service} {status.version || ''}</div>
    </div>
  );
}
