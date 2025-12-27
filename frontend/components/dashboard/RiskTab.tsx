/**
 * Risk & Safety Tab Component
 * DASHBOARD-V3-001: Full Visual UI
 * 
 * Displays:
 * - RiskGate decisions breakdown
 * - ESS triggers timeline
 * - Drawdown per profile
 * - VaR/ES snapshot
 */

import { useEffect, useState } from 'react';
import DashboardCard from '../DashboardCard';

interface RiskData {
  timestamp: string;
  risk_gate_decisions_stats: {
    allow: number;
    block: number;
    scale: number;
    total: number;
  };
  ess_triggers_recent: any[];
  dd_per_profile: any[];
  var_es_snapshot: {
    var_95: number;
    var_99: number;
    es_95: number;
    es_99: number;
  };
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function RiskTab() {
  const [data, setData] = useState<RiskData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchRiskData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/dashboard/risk`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const json = await response.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch risk data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchRiskData();
    // Poll every 10 seconds
    const interval = setInterval(fetchRiskData, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {[1, 2, 3, 4].map(i => (
          <div key={i} className="dashboard-card h-64 animate-pulse bg-gray-200" />
        ))}
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="text-center py-12">
        <p className="text-danger text-lg">⚠️ {error || 'No data'}</p>
        <button onClick={fetchRiskData} className="mt-4 px-4 py-2 bg-primary text-white rounded-lg">
          Retry
        </button>
      </div>
    );
  }

  const { risk_gate_decisions_stats, ess_triggers_recent, dd_per_profile, var_es_snapshot } = data;

  // Calculate percentages
  const total = risk_gate_decisions_stats.total || 1;
  const allowPct = ((risk_gate_decisions_stats.allow / total) * 100).toFixed(0);
  const blockPct = ((risk_gate_decisions_stats.block / total) * 100).toFixed(0);
  const scalePct = ((risk_gate_decisions_stats.scale / total) * 100).toFixed(0);

  return (
    <div className="space-y-6">
      {/* RiskGate Decisions Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <DashboardCard title="Total Decisions">
          <div className="text-center py-4">
            <div className="text-4xl font-bold text-gray-900 dark:text-white">
              {risk_gate_decisions_stats.total}
            </div>
            <p className="mt-2 text-xs text-gray-600 dark:text-gray-400">
              RiskGate evaluations
            </p>
          </div>
        </DashboardCard>

        <DashboardCard title="Allowed">
          <div className="text-center py-4">
            <div className="text-4xl font-bold text-success">
              {risk_gate_decisions_stats.allow}
            </div>
            <p className="mt-2 text-sm text-success font-semibold">
              {allowPct}%
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Passed risk checks
            </p>
          </div>
        </DashboardCard>

        <DashboardCard title="Scaled">
          <div className="text-center py-4">
            <div className="text-4xl font-bold text-warning">
              {risk_gate_decisions_stats.scale}
            </div>
            <p className="mt-2 text-sm text-warning font-semibold">
              {scalePct}%
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Position size reduced
            </p>
          </div>
        </DashboardCard>

        <DashboardCard title="Blocked">
          <div className="text-center py-4">
            <div className="text-4xl font-bold text-danger">
              {risk_gate_decisions_stats.block}
            </div>
            <p className="mt-2 text-sm text-danger font-semibold">
              {blockPct}%
            </p>
            <p className="text-xs text-gray-600 dark:text-gray-400">
              Trade rejected
            </p>
          </div>
        </DashboardCard>
      </div>

      {/* ESS Triggers & DD per Profile */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ESS Triggers Recent */}
        <DashboardCard title="ESS Triggers (Last 24h)">
          <div className="p-4">
            {ess_triggers_recent.length > 0 ? (
              <div className="space-y-3 max-h-80 overflow-y-auto">
                {ess_triggers_recent.map((trigger, idx) => (
                  <div key={idx} className="border border-danger rounded-lg p-3 bg-red-50 dark:bg-red-900/20">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-xs bg-danger text-white px-2 py-1 rounded">
                        TRIGGERED
                      </span>
                      <span className="text-xs text-gray-600 dark:text-gray-400">
                        {new Date(trigger.timestamp).toLocaleString()}
                      </span>
                    </div>
                    <p className="text-sm font-semibold text-danger">
                      Reason: {trigger.reason || 'Daily loss threshold exceeded'}
                    </p>
                    <p className="text-xs text-gray-700 dark:text-gray-300 mt-1">
                      Daily Loss: {trigger.daily_loss?.toFixed(2)}%
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-center py-12">
                <div className="text-success text-5xl mb-2">✅</div>
                <p className="text-gray-600 dark:text-gray-400">
                  No ESS triggers in last 24h
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  System operating normally
                </p>
              </div>
            )}
          </div>
        </DashboardCard>

        {/* Drawdown per Profile */}
        <DashboardCard title="Drawdown per Profile">
          <div className="p-4">
            {dd_per_profile.length > 0 ? (
              <div className="space-y-3">
                {dd_per_profile.map((profile, idx) => (
                  <div key={idx} className="border border-gray-200 dark:border-gray-700 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-semibold text-gray-900 dark:text-white">
                        {profile.profile_name || 'Unknown'}
                      </span>
                      <span className={`text-sm font-bold ${
                        (profile.current_dd || 0) <= -5 ? 'text-danger' :
                        (profile.current_dd || 0) <= -2 ? 'text-warning' :
                        'text-success'
                      }`}>
                        {profile.current_dd?.toFixed(2)}%
                      </span>
                    </div>
                    <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          (profile.current_dd || 0) <= -5 ? 'bg-danger' :
                          (profile.current_dd || 0) <= -2 ? 'bg-warning' :
                          'bg-success'
                        }`}
                        style={{ width: `${Math.abs(profile.current_dd || 0) * 10}%` }}
                      />
                    </div>
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      Max DD: {profile.max_dd?.toFixed(2)}% | Limit: {profile.limit?.toFixed(2)}%
                    </p>
                  </div>
                ))}
              </div>
            ) : (
              <p className="text-center text-gray-600 dark:text-gray-400 py-12">
                No drawdown data available
              </p>
            )}
          </div>
        </DashboardCard>
      </div>

      {/* VaR / ES Snapshot */}
      <DashboardCard title="Value at Risk (VaR) & Expected Shortfall (ES)">
        <div className="p-4">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">VaR 95%</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ${var_es_snapshot.var_95.toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 mt-1">95% confidence</p>
            </div>

            <div className="text-center">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">VaR 99%</p>
              <p className="text-2xl font-bold text-danger">
                ${var_es_snapshot.var_99.toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 mt-1">99% confidence</p>
            </div>

            <div className="text-center">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">ES 95%</p>
              <p className="text-2xl font-bold text-gray-900 dark:text-white">
                ${var_es_snapshot.es_95.toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 mt-1">Tail risk</p>
            </div>

            <div className="text-center">
              <p className="text-xs text-gray-600 dark:text-gray-400 mb-1">ES 99%</p>
              <p className="text-2xl font-bold text-danger">
                ${var_es_snapshot.es_99.toFixed(2)}
              </p>
              <p className="text-xs text-gray-500 mt-1">Extreme loss</p>
            </div>
          </div>

          <div className="mt-4 p-3 bg-blue-50 dark:bg-blue-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <p className="text-xs text-gray-700 dark:text-gray-300">
              <strong>VaR:</strong> Maximum expected loss under normal conditions at given confidence level.<br />
              <strong>ES:</strong> Average loss when VaR threshold is exceeded (tail risk measure).
            </p>
          </div>
        </div>
      </DashboardCard>

      {/* Update Timestamp */}
      <div className="text-center text-xs text-gray-500 dark:text-gray-400">
        Last updated: {new Date(data.timestamp).toLocaleString()}
      </div>
    </div>
  );
}
