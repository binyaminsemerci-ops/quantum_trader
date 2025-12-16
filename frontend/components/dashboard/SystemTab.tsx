/**
 * System & Stress Tab Component
 * DASHBOARD-V3-001: Full Visual UI
 * 
 * Displays:
 * - Microservices health grid
 * - Exchanges health
 * - Failover events
 * - Stress scenarios with run button
 */

import { useEffect, useState } from 'react';
import DashboardCard from '../DashboardCard';

interface SystemData {
  timestamp: string;
  services_health: Array<{ name: string; status: string; latency?: number }>;
  exchanges_health: Array<{ exchange: string; status: string; latency?: number }>;
  failover_events_recent: any[];
  stress_scenarios_recent: any[];
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function SystemTab() {
  const [data, setData] = useState<SystemData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [runningStress, setRunningStress] = useState(false);
  const [stressResult, setStressResult] = useState<string | null>(null);

  const fetchSystemData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/dashboard/system`);
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      const json = await response.json();
      setData(json);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch system data');
    } finally {
      setLoading(false);
    }
  };

  const runAllStressScenarios = async () => {
    setRunningStress(true);
    setStressResult(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/dashboard/stress/run_all`, {
        method: 'POST'
      });
      
      if (!response.ok) throw new Error(`HTTP ${response.status}`);
      
      const result = await response.json();
      setStressResult(`✅ Stress scenarios completed: ${result.passed || 0} passed, ${result.failed || 0} failed`);
      
      // Refresh system data after stress test
      await fetchSystemData();
    } catch (err) {
      setStressResult(`❌ Error: ${err instanceof Error ? err.message : 'Failed to run stress scenarios'}`);
    } finally {
      setRunningStress(false);
    }
  };

  useEffect(() => {
    fetchSystemData();
    // Poll every 15 seconds
    const interval = setInterval(fetchSystemData, 15000);
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
        <button onClick={fetchSystemData} className="mt-4 px-4 py-2 bg-primary text-white rounded-lg">
          Retry
        </button>
      </div>
    );
  }

  const { services_health, exchanges_health, failover_events_recent, stress_scenarios_recent } = data;

  return (
    <div className="space-y-6">
      {/* Microservices Health Grid */}
      <DashboardCard title="Microservices Health">
        <div className="p-4 grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {services_health.length > 0 ? (
            services_health.map((service, idx) => (
              <div
                key={idx}
                className={`p-4 rounded-lg border-2 ${
                  service.status === 'UP' || service.status === 'OK' || service.status === 'HEALTHY'
                    ? 'border-success bg-green-50 dark:bg-green-900/20'
                    : service.status === 'DEGRADED'
                    ? 'border-warning bg-yellow-50 dark:bg-yellow-900/20'
                    : 'border-danger bg-red-50 dark:bg-red-900/20'
                }`}
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-gray-700 dark:text-gray-300">
                    {service.name}
                  </span>
                  <span className={`w-3 h-3 rounded-full ${
                    service.status === 'UP' || service.status === 'OK' || service.status === 'HEALTHY'
                      ? 'bg-success'
                      : service.status === 'DEGRADED'
                      ? 'bg-warning'
                      : 'bg-danger'
                  }`} />
                </div>
                <p className={`text-sm font-semibold ${
                  service.status === 'UP' || service.status === 'OK' || service.status === 'HEALTHY'
                    ? 'text-success'
                    : service.status === 'DEGRADED'
                    ? 'text-warning'
                    : 'text-danger'
                }`}>
                  {service.status}
                </p>
                {service.latency !== undefined && (
                  <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                    {service.latency}ms
                  </p>
                )}
              </div>
            ))
          ) : (
            <p className="col-span-full text-center text-gray-600 dark:text-gray-400 py-8">
              No service health data
            </p>
          )}
        </div>
      </DashboardCard>

      {/* Exchanges Health & Failover Events */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Exchanges Health */}
        <DashboardCard title="Exchanges Health">
          <div className="p-4 space-y-3">
            {exchanges_health.length > 0 ? (
              exchanges_health.map((exchange, idx) => (
                <div
                  key={idx}
                  className={`p-3 rounded-lg border ${
                    exchange.status === 'UP' || exchange.status === 'HEALTHY'
                      ? 'border-success bg-green-50 dark:bg-green-900/20'
                      : exchange.status === 'DEGRADED'
                      ? 'border-warning bg-yellow-50 dark:bg-yellow-900/20'
                      : 'border-danger bg-red-50 dark:bg-red-900/20'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-semibold text-gray-900 dark:text-white">
                        {exchange.exchange}
                      </p>
                      {exchange.latency !== undefined && (
                        <p className="text-xs text-gray-600 dark:text-gray-400">
                          Latency: {exchange.latency}ms
                        </p>
                      )}
                    </div>
                    <span className={`px-3 py-1 rounded text-xs font-bold ${
                      exchange.status === 'UP' || exchange.status === 'HEALTHY'
                        ? 'bg-success text-white'
                        : exchange.status === 'DEGRADED'
                        ? 'bg-warning text-white'
                        : 'bg-danger text-white'
                    }`}>
                      {exchange.status}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-center text-gray-600 dark:text-gray-400 py-8">
                No exchange health data
              </p>
            )}
          </div>
        </DashboardCard>

        {/* Failover Events */}
        <DashboardCard title="Failover Events (Recent)">
          <div className="p-4 space-y-3 max-h-80 overflow-y-auto">
            {failover_events_recent.length > 0 ? (
              failover_events_recent.map((event, idx) => (
                <div key={idx} className="border border-warning rounded-lg p-3 bg-yellow-50 dark:bg-yellow-900/20">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-xs bg-warning text-white px-2 py-1 rounded">
                      FAILOVER
                    </span>
                    <span className="text-xs text-gray-600 dark:text-gray-400">
                      {new Date(event.timestamp).toLocaleString()}
                    </span>
                  </div>
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">
                    {event.from_exchange} → {event.to_exchange}
                  </p>
                  <p className="text-xs text-gray-700 dark:text-gray-300 mt-1">
                    Reason: {event.reason || 'Exchange unavailable'}
                  </p>
                </div>
              ))
            ) : (
              <div className="text-center py-12">
                <div className="text-success text-5xl mb-2">✅</div>
                <p className="text-gray-600 dark:text-gray-400">
                  No recent failover events
                </p>
                <p className="text-xs text-gray-500 mt-2">
                  All exchanges operating normally
                </p>
              </div>
            )}
          </div>
        </DashboardCard>
      </div>

      {/* Stress Scenarios */}
      <DashboardCard title="Stress Test Scenarios">
        <div className="p-4">
          {/* Run Stress Button - Phase 10: Enhanced styling */}
          <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 dark:from-blue-900/20 dark:to-indigo-900/20 rounded-lg border border-blue-200 dark:border-blue-800">
            <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div>
                <h3 className="font-semibold text-gray-900 dark:text-white text-lg">
                  🧪 Run All Stress Scenarios
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                  Tests: Flash Crash, Exchange Outage, Network Latency, High Volatility
                </p>
              </div>
              <button
                onClick={runAllStressScenarios}
                disabled={runningStress}
                className={`px-6 py-3 rounded-lg font-semibold transition-all duration-200 whitespace-nowrap ${
                  runningStress
                    ? 'bg-gray-400 text-white cursor-not-allowed opacity-50'
                    : 'bg-gradient-to-r from-primary to-blue-600 text-white hover:shadow-lg hover:scale-105 focus:outline-none focus:ring-2 focus:ring-primary focus:ring-offset-2'
                }`}
              >
                {runningStress ? (
                  <>
                    <span className="inline-block animate-spin mr-2">⏳</span>
                    Running...
                  </>
                ) : (
                  <>▶️ Run All Scenarios</>
                )}
              </button>
            </div>
          </div>

          {/* Stress Result - Phase 10: Improved styling */}
          {stressResult && (
            <div className={`p-4 rounded-lg mb-6 border-2 ${
              stressResult.startsWith('✅')
                ? 'bg-green-50 dark:bg-green-900/20 border-success'
                : 'bg-red-50 dark:bg-red-900/20 border-danger'
            }`}>
              <p className={`text-sm font-semibold ${
                stressResult.startsWith('✅') ? 'text-success' : 'text-danger'
              }`}>
                {stressResult}
              </p>
            </div>
          )}

          {/* Recent Stress Runs */}
          <div className="space-y-3 max-h-80 overflow-y-auto">
            <h4 className="font-semibold text-gray-700 dark:text-gray-300 mb-2">
              Recent Runs
            </h4>
            {stress_scenarios_recent.length > 0 ? (
              stress_scenarios_recent.map((run, idx) => (
                <div key={idx} className={`border rounded-lg p-3 ${
                  run.success
                    ? 'border-success bg-green-50 dark:bg-green-900/20'
                    : 'border-danger bg-red-50 dark:bg-red-900/20'
                }`}>
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-semibold text-gray-900 dark:text-white">
                      {run.scenario_name}
                    </span>
                    <div className="flex items-center space-x-2">
                      <span className={`px-2 py-1 rounded text-xs font-bold ${
                        run.success ? 'bg-success text-white' : 'bg-danger text-white'
                      }`}>
                        {run.success ? 'PASS' : 'FAIL'}
                      </span>
                      <span className="text-xs text-gray-600 dark:text-gray-400">
                        {new Date(run.timestamp).toLocaleTimeString()}
                      </span>
                    </div>
                  </div>
                  {run.reason && (
                    <p className="text-xs text-gray-700 dark:text-gray-300">
                      {run.reason}
                    </p>
                  )}
                  {run.duration_ms && (
                    <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                      Duration: {run.duration_ms}ms
                    </p>
                  )}
                </div>
              ))
            ) : (
              <p className="text-center text-gray-600 dark:text-gray-400 py-8">
                No recent stress test runs
              </p>
            )}
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
