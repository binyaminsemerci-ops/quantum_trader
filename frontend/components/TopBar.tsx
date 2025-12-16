// TopBar - ESS status, system status, last update
// Phase 10: Enhanced with environment badge and improved styling
import type { ESSState, ServiceStatus } from '@/lib/types';
import { getESSStateColor, getServiceStatusColor, formatRelativeTime } from '@/lib/utils';
import { useDashboardStream } from '@/hooks/useDashboardStream';

interface TopBarProps {
  essState: ESSState;
  systemStatus: ServiceStatus;
  lastUpdate: string | null;
  wsConnected: boolean;
}

export default function TopBar({ essState, systemStatus, lastUpdate, wsConnected }: TopBarProps) {
  // [PHASE 9] Real-time position count
  const { data: streamData } = useDashboardStream();
  
  // Detect environment from ENV or default to STAGING
  const environment = process.env.NEXT_PUBLIC_ENVIRONMENT || 'STAGING';
  const isProduction = environment === 'PRODUCTION';
  
  return (
    <div className="bg-white dark:bg-slate-800 border-b border-gray-200 dark:border-slate-700 px-4 sm:px-6 py-3 shadow-sm">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 sm:gap-4">
        {/* Left: Title + Environment Badge */}
        <div className="flex items-center gap-3">
          <div>
            <h1 className="text-xl sm:text-2xl font-bold text-gray-900 dark:text-white">
              Quantum Trader
            </h1>
            <p className="text-xs sm:text-sm text-gray-600 dark:text-gray-400">
              Live Trading Dashboard
            </p>
          </div>
          
          {/* Environment Badge */}
          <span className={`px-2.5 py-1 rounded-md text-xs font-bold uppercase ${
            isProduction 
              ? 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-300' 
              : 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-300'
          }`}>
            {isProduction ? '🔴 PROD' : '⚡ TESTNET'}
          </span>
        </div>

        {/* Right: Status badges */}
        <div className="flex flex-wrap items-center gap-2 sm:gap-4 text-sm">
          {/* [PHASE 9] Live position count */}
          {streamData && (
            <div className="flex items-center gap-2 px-3 py-1.5 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
              <span className="text-sm font-semibold text-blue-700 dark:text-blue-300">
                📊 {streamData.open_positions_count}
              </span>
              <span className="text-xs text-blue-600 dark:text-blue-400">
                positions
              </span>
            </div>
          )}
          
          {/* WebSocket status */}
          <div className="flex items-center gap-2 px-2.5 py-1.5 bg-gray-50 dark:bg-slate-700/50 rounded-lg">
            <div className={`w-2 h-2 rounded-full ${
              wsConnected ? 'bg-success animate-pulse' : 'bg-gray-400'
            }`} />
            <span className="text-xs font-medium text-gray-700 dark:text-gray-300">
              {wsConnected ? 'Live' : 'Offline'}
            </span>
          </div>

          {/* Last update */}
          {lastUpdate && (
            <div className="hidden sm:flex items-center text-xs text-gray-600 dark:text-gray-400">
              🕐 {formatRelativeTime(lastUpdate)}
            </div>
          )}

          {/* System status */}
          <div className="flex items-center gap-2">
            <span className="hidden sm:inline text-xs text-gray-600 dark:text-gray-400">
              System:
            </span>
            <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
              getServiceStatusColor(systemStatus)
            }`}>
              {systemStatus}
            </span>
          </div>

          {/* ESS status */}
          <div className="flex items-center gap-2">
            <span className="hidden sm:inline text-xs text-gray-600 dark:text-gray-400">
              ESS:
            </span>
            <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${
              getESSStateColor(essState)
            }`}>
              {essState}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
