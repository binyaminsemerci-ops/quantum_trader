/**
 * Dashboard Auto-Repair Control Panel
 * UI komponent for å kontrollere automatisk reparasjon
 */

import { useState } from 'react';
import { useDashboardAutoRepair } from '../hooks/useDashboardAutoRepair';

interface DashboardAutoRepairPanelProps {
  className?: string;
}

export default function DashboardAutoRepairPanel({ className = '' }: DashboardAutoRepairPanelProps): JSX.Element {
  const [isOpen, setIsOpen] = useState(false);
  const [showLog, setShowLog] = useState(false);
  
  const {
    isHealthy,
    issuesFound,
    criticalIssues,
    autoRepairActive,
    lastCheck,
    healthReport,
    repairLog,
    performHealthCheck,
    manualRepair,
    resetToOptimal,
    clearRepairLog,
    toggleAutoRepair,
    config
  } = useDashboardAutoRepair();

  const getHealthColor = () => {
    if (criticalIssues > 0) return 'text-red-500';
    if (issuesFound > 0) return 'text-yellow-500';
    return 'text-green-500';
  };

  const getHealthIcon = () => {
    if (criticalIssues > 0) return '🚨';
    if (issuesFound > 0) return '⚠️';
    return '✅';
  };

  const getHealthMessage = () => {
    if (criticalIssues > 0) return `${criticalIssues} critical issues found`;
    if (issuesFound > 0) return `${issuesFound} minor issues found`;
    return 'Dashboard is healthy';
  };

  return (
    <div className={`relative ${className}`}>
      {/* Health Status Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className={`
          flex items-center space-x-2 px-3 py-2 rounded-lg font-medium transition-all
          ${criticalIssues > 0 
            ? 'bg-red-100 hover:bg-red-200 text-red-700 border border-red-300' 
            : issuesFound > 0 
              ? 'bg-yellow-100 hover:bg-yellow-200 text-yellow-700 border border-yellow-300'
              : 'bg-green-100 hover:bg-green-200 text-green-700 border border-green-300'
          }
          ${autoRepairActive ? 'animate-pulse' : ''}
        `}
        title="Dashboard Health & Auto-Repair"
      >
        <span className="text-lg">{getHealthIcon()}</span>
        <span className="hidden sm:inline">
          {autoRepairActive ? 'Repairing...' : 'Auto-Repair'}
        </span>
        {issuesFound > 0 && (
          <span className="bg-white rounded-full px-2 py-1 text-xs font-bold">
            {issuesFound}
          </span>
        )}
      </button>

      {/* Control Panel */}
      {isOpen && (
        <div className="absolute right-0 top-full mt-2 w-96 bg-white dark:bg-gray-800 rounded-xl shadow-xl border dark:border-gray-700 p-4 z-50">
          
          {/* Header */}
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-bold text-gray-900 dark:text-gray-100">
              🤖 Dashboard Auto-Repair
            </h3>
            <button
              onClick={() => setIsOpen(false)}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              ✕
            </button>
          </div>

          {/* Health Status */}
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-3 mb-4">
            <div className="flex items-center justify-between mb-2">
              <span className="font-medium text-gray-700 dark:text-gray-300">Health Status</span>
              <span className={`font-bold ${getHealthColor()}`}>
                {getHealthIcon()} {getHealthMessage()}
              </span>
            </div>
            
            {lastCheck && (
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Last checked: {lastCheck.toLocaleTimeString()}
              </div>
            )}

            {healthReport && healthReport.recommendations.length > 0 && (
              <div className="mt-2">
                <div className="text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
                  Recommendations:
                </div>
                {healthReport.recommendations.map((rec, index) => (
                  <div key={index} className="text-xs text-gray-500 dark:text-gray-400">
                    • {rec}
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Quick Actions */}
          <div className="space-y-2 mb-4">
            <button
              onClick={performHealthCheck}
              disabled={autoRepairActive}
              className="w-full px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-all"
            >
              🔍 Run Health Check
            </button>
            
            <button
              onClick={manualRepair}
              disabled={autoRepairActive || issuesFound === 0}
              className="w-full px-3 py-2 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-all"
            >
              🔧 Manual Repair ({issuesFound} issues)
            </button>
            
            <button
              onClick={resetToOptimal}
              disabled={autoRepairActive}
              className="w-full px-3 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white rounded-lg font-medium transition-all"
            >
              🎯 Reset to Optimal Layout
            </button>
          </div>

          {/* Auto-Repair Settings */}
          <div className="border-t dark:border-gray-600 pt-3 mb-4">
            <label className="flex items-center justify-between">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Auto-Repair Enabled
              </span>
              <input
                type="checkbox"
                checked={config.enabled}
                onChange={(e) => toggleAutoRepair(e.target.checked)}
                className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
              />
            </label>
            
            <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Automatically fixes critical issues when detected
            </div>
          </div>

          {/* Repair Log */}
          <div className="border-t dark:border-gray-600 pt-3">
            <div className="flex items-center justify-between mb-2">
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                Activity Log ({repairLog.length})
              </span>
              <div className="flex space-x-2">
                <button
                  onClick={() => setShowLog(!showLog)}
                  className="text-xs text-blue-600 hover:text-blue-700"
                >
                  {showLog ? 'Hide' : 'Show'}
                </button>
                <button
                  onClick={clearRepairLog}
                  className="text-xs text-red-600 hover:text-red-700"
                >
                  Clear
                </button>
              </div>
            </div>

            {showLog && repairLog.length > 0 && (
              <div className="bg-gray-900 text-green-400 text-xs p-2 rounded max-h-32 overflow-y-auto font-mono">
                {repairLog.slice(-10).map((entry, index) => (
                  <div key={index} className="mb-1">
                    {entry}
                  </div>
                ))}
              </div>
            )}

            {showLog && repairLog.length === 0 && (
              <div className="text-xs text-gray-500 dark:text-gray-400 italic">
                No activity yet
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="text-xs text-gray-400 dark:text-gray-500 mt-3 pt-3 border-t dark:border-gray-600">
            🤖 AI-powered dashboard health monitoring
          </div>
        </div>
      )}
    </div>
  );
}