import { useState, useEffect } from "react";
import { api, AiOsStatus, AiOsModuleHealth } from "../lib/api";

export default function AiOsStatusWidget() {
  const [status, setStatus] = useState<AiOsStatus | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const data = await api.getAiOsStatus();
        setStatus(data);
        setLoading(false);
      } catch (error) {
        console.error("Failed to fetch AI-OS status:", error);
        setLoading(false);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000); // Refresh every 5 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="card p-4 shadow-card">
        <h3 className="text-sm font-medium text-slate-600 dark:text-slate-300 mb-4">
          AI-OS System Health
        </h3>
        <div className="flex items-center justify-center h-32">
          <span className="text-slate-400">Loading...</span>
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="card p-4 shadow-card">
        <h3 className="text-sm font-medium text-slate-600 dark:text-slate-300 mb-4">
          AI-OS System Health
        </h3>
        <div className="flex items-center justify-center h-32">
          <span className="text-slate-400">No data available</span>
        </div>
      </div>
    );
  }

  const getHealthColor = (health: string) => {
    switch (health.toUpperCase()) {
      case "HEALTHY":
      case "OPTIMAL":
        return "text-emerald-600";
      case "DEGRADED":
      case "WARNING":
        return "text-yellow-600";
      case "CRITICAL":
      case "EMERGENCY":
        return "text-red-600";
      default:
        return "text-slate-500";
    }
  };

  const getHealthDot = (health: string) => {
    switch (health.toUpperCase()) {
      case "HEALTHY":
      case "OPTIMAL":
        return "bg-emerald-500";
      case "DEGRADED":
      case "WARNING":
        return "bg-yellow-500";
      case "CRITICAL":
      case "EMERGENCY":
        return "bg-red-500";
      default:
        return "bg-slate-400";
    }
  };

  const getRiskModeColor = (mode: string) => {
    switch (mode.toUpperCase()) {
      case "SAFE":
        return "bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300";
      case "NORMAL":
        return "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-300";
      case "AGGRESSIVE":
        return "bg-orange-100 text-orange-800 dark:bg-orange-900/30 dark:text-orange-300";
      case "HEDGEFUND":
        return "bg-purple-100 text-purple-800 dark:bg-purple-900/30 dark:text-purple-300";
      default:
        return "bg-slate-100 text-slate-800 dark:bg-slate-900/30 dark:text-slate-300";
    }
  };

  return (
    <div className="card p-4 shadow-card">
      <h3 className="text-sm font-medium text-slate-600 dark:text-slate-300 mb-4">
        AI-OS System Health
      </h3>

      {/* Overall Status & Key Indicators */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="p-3 rounded-lg bg-black/5 dark:bg-white/5">
          <div className="text-xs text-slate-500 mb-1">Overall Health</div>
          <div className={`text-lg font-bold ${getHealthColor(status.overall_health)}`}>
            {status.overall_health}
          </div>
        </div>
        <div className="p-3 rounded-lg bg-black/5 dark:bg-white/5">
          <div className="text-xs text-slate-500 mb-1">Risk Mode</div>
          <div className={`inline-block px-2 py-1 rounded text-xs font-semibold ${getRiskModeColor(status.risk_mode)}`}>
            {status.risk_mode}
          </div>
        </div>
      </div>

      {/* Emergency Brake & Trades Status */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${status.emergency_brake ? "bg-red-500" : "bg-slate-300"}`} />
          <span className="text-xs text-slate-600 dark:text-slate-400">
            Emergency Brake: {status.emergency_brake ? "ACTIVE" : "OFF"}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${status.new_trades_allowed ? "bg-emerald-500" : "bg-slate-300"}`} />
          <span className="text-xs text-slate-600 dark:text-slate-400">
            New Trades: {status.new_trades_allowed ? "ALLOWED" : "BLOCKED"}
          </span>
        </div>
      </div>

      {/* Subsystems Table */}
      <div className="border-t border-slate-200 dark:border-slate-700 pt-3">
        <div className="text-xs font-medium text-slate-500 mb-2">Subsystems</div>
        <div className="space-y-1 max-h-48 overflow-y-auto">
          {status.modules.map((module: AiOsModuleHealth, idx: number) => (
            <div
              key={idx}
              className="flex items-center gap-2 p-2 rounded hover:bg-black/5 dark:hover:bg-white/5 transition-colors"
            >
              <div className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${getHealthDot(module.health)}`} />
              <div className="flex-1 min-w-0">
                <div className="text-xs font-medium text-slate-700 dark:text-slate-300 truncate">
                  {module.name}
                </div>
                <div className="text-xs text-slate-500 truncate">
                  {module.note}
                </div>
              </div>
              <div className={`text-xs font-semibold ${getHealthColor(module.health)}`}>
                {module.health}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
