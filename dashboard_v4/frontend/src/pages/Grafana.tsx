import { useState } from 'react';

const GRAFANA_BASE = '/grafana';

interface DashboardInfo {
  uid: string;
  title: string;
  category: string;
  icon: string;
}

const DASHBOARDS: DashboardInfo[] = [
  { uid: 'sys-overview-clean', title: 'System Overview — Systemd', category: 'Infrastructure', icon: '🖥️' },
  { uid: 'infra-clean', title: 'Infrastructure — Systemd', category: 'Infrastructure', icon: '⚙️' },
  { uid: 'services-clean', title: 'Quantum Services', category: 'Infrastructure', icon: '🔧' },
  { uid: 'redis-clean', title: 'Redis Metrics', category: 'Infrastructure', icon: '🔴' },
  { uid: 'logs-clean', title: 'Log Aggregation', category: 'Logs', icon: '📝' },
  { uid: 'p1b-logs', title: 'P1-B: Log Aggregation', category: 'Logs', icon: '📋' },
  { uid: 'p1c-baseline', title: 'P1-C: Performance Baseline', category: 'Performance', icon: '📈' },
  { uid: 'rl-shadow-clean', title: 'RL Shadow Monitoring', category: 'AI / RL', icon: '🧠' },
  { uid: 'rl-shadow-performance', title: 'RL Shadow Performance', category: 'AI / RL', icon: '🎯' },
  { uid: 'b558bc08-73cf-4e97-80c2-6c0eb37aa564', title: 'Execution & Trading', category: 'Trading', icon: '💹' },
  { uid: 'eff09fa9-6865-4a9e-b9d3-3568bb61b682', title: 'Harvest Control (P2.7)', category: 'Trading', icon: '🌾' },
  { uid: '1079a1b3-bc66-4d49-bafe-4a388f78f4de', title: 'Harvest Optimizer (P3.9)', category: 'Trading', icon: '🎯' },
  { uid: 'b3a96f3a-1ceb-480f-a7a7-c3b6f044dca7', title: 'Safety Telemetry (P1)', category: 'Safety', icon: '🛡️' },
];

type TabId = 'overview' | 'logs' | 'dashboards';

export default function Grafana() {
  const [activeTab, setActiveTab] = useState<TabId>('overview');
  const [selectedDashboard, setSelectedDashboard] = useState<string>('sys-overview-clean');

  const tabs: { id: TabId; label: string; icon: string }[] = [
    { id: 'overview', label: 'System Overview', icon: '🖥️' },
    { id: 'logs', label: 'Log Analysis', icon: '📝' },
    { id: 'dashboards', label: 'All Dashboards', icon: '📊' },
  ];

  const categories = [...new Set(DASHBOARDS.map(d => d.category))];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-purple-400">Grafana Monitoring</h1>
          <p className="text-gray-400 mt-2">Real-time metrics and log analysis dashboards</p>
        </div>
        <a
          href={`${GRAFANA_BASE}/`}
          target="_blank"
          rel="noopener noreferrer"
          className="px-4 py-2 bg-purple-500 hover:bg-purple-600 rounded-lg text-white font-medium transition-colors"
        >
          Open Full Grafana ↗
        </a>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-2 bg-gray-800 rounded-lg p-1">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`flex-1 px-6 py-3 rounded-lg font-medium transition-all ${
              activeTab === tab.id
                ? 'bg-purple-500 text-white shadow-lg'
                : 'text-gray-400 hover:text-white hover:bg-gray-700'
            }`}
          >
            {tab.icon} {tab.label}
          </button>
        ))}
      </div>

      {/* ── System Overview Tab ── */}
      {activeTab === 'overview' && (
        <div className="space-y-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h2 className="text-2xl font-semibold text-white">Quantum System Overview</h2>
                <p className="text-gray-400 mt-1">Comprehensive system metrics across all services</p>
              </div>
              <a
                href={`${GRAFANA_BASE}/d/sys-overview-clean`}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg text-white text-sm font-medium transition-colors"
              >
                Full Screen ↗
              </a>
            </div>

            <div className="bg-gray-900 rounded-lg overflow-hidden shadow-2xl h-[900px]">
              <iframe
                src={`${GRAFANA_BASE}/d/sys-overview-clean/system-overview-systemd?kiosk=tv&refresh=30s&from=now-6h&to=now`}
                width="100%"
                height="100%"
                frameBorder="0"
                title="Quantum System Overview"
                className="w-full h-full"
              />
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-gray-400 mb-1">Dashboard</div>
                <div className="text-white font-semibold">System Overview — Systemd</div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-gray-400 mb-1">Auto-Refresh</div>
                <div className="text-green-400 font-semibold">Every 30 seconds</div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-gray-400 mb-1">Time Range</div>
                <div className="text-blue-400 font-semibold">Last 6 hours</div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-gray-400 mb-1">Data Source</div>
                <div className="text-purple-400 font-semibold">Prometheus</div>
              </div>
            </div>
          </div>

          {/* Quick Links to Related Dashboards */}
          <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">📊 Related Dashboards</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <div className="text-purple-400 font-bold mb-2">System Resources</div>
                <ul className="text-sm space-y-2">
                  <li>
                    <a href={`${GRAFANA_BASE}/d/sys-overview-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-purple-400 transition-colors flex items-center gap-1">
                      🖥️ System Overview <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                  <li>
                    <a href={`${GRAFANA_BASE}/d/infra-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-purple-400 transition-colors flex items-center gap-1">
                      ⚙️ Infrastructure <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                  <li>
                    <a href={`${GRAFANA_BASE}/d/services-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-purple-400 transition-colors flex items-center gap-1">
                      🔧 Service Status <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <div className="text-blue-400 font-bold mb-2">Services</div>
                <ul className="text-sm space-y-2">
                  <li>
                    <a href={`${GRAFANA_BASE}/d/services-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-blue-400 transition-colors flex items-center gap-1">
                      📋 Running Services <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                  <li>
                    <a href={`${GRAFANA_BASE}/d/p1c-baseline`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-blue-400 transition-colors flex items-center gap-1">
                      📈 Performance <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                  <li>
                    <a href={`${GRAFANA_BASE}/d/rl-shadow-performance`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-blue-400 transition-colors flex items-center gap-1">
                      🧠 RL Performance <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <div className="text-green-400 font-bold mb-2">Redis & Data</div>
                <ul className="text-sm space-y-2">
                  <li>
                    <a href={`${GRAFANA_BASE}/d/redis-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-green-400 transition-colors flex items-center gap-1">
                      🔴 Redis Metrics <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                  <li>
                    <a href={`${GRAFANA_BASE}/d/c587ba0d-7b88-486b-a03e-6ff436dd65c2`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-green-400 transition-colors flex items-center gap-1">
                      🐘 Redis & Postgres <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <div className="text-yellow-400 font-bold mb-2">Safety & Logs</div>
                <ul className="text-sm space-y-2">
                  <li>
                    <a href={`${GRAFANA_BASE}/d/b3a96f3a-1ceb-480f-a7a7-c3b6f044dca7`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-yellow-400 transition-colors flex items-center gap-1">
                      🛡️ Safety Telemetry <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                  <li>
                    <a href={`${GRAFANA_BASE}/d/logs-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-yellow-400 transition-colors flex items-center gap-1">
                      📝 Log Aggregation <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── Log Analysis Tab ── */}
      {activeTab === 'logs' && (
        <div className="space-y-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h2 className="text-2xl font-semibold text-white">Log Aggregation</h2>
                <p className="text-gray-400 mt-1">Centralized log collection with error tracking and analysis</p>
              </div>
              <a
                href={`${GRAFANA_BASE}/d/logs-clean`}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg text-white text-sm font-medium transition-colors"
              >
                Full Screen ↗
              </a>
            </div>

            <div className="bg-gray-900 rounded-lg overflow-hidden shadow-2xl h-[900px]">
              <iframe
                src={`${GRAFANA_BASE}/d/logs-clean/log-aggregation?kiosk=tv&refresh=30s&from=now-24h&to=now`}
                width="100%"
                height="100%"
                frameBorder="0"
                title="Log Aggregation"
                className="w-full h-full"
              />
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-gray-400 mb-1">Dashboard</div>
                <div className="text-white font-semibold">Log Aggregation</div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-gray-400 mb-1">Auto-Refresh</div>
                <div className="text-green-400 font-semibold">Every 30 seconds</div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-gray-400 mb-1">Time Range</div>
                <div className="text-blue-400 font-semibold">Last 24 hours</div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-gray-400 mb-1">Data Source</div>
                <div className="text-purple-400 font-semibold">Loki</div>
              </div>
            </div>
          </div>

          {/* Log-Related Dashboards */}
          <div className="bg-gradient-to-r from-purple-500/20 to-red-500/20 border border-purple-500/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">📝 Log Dashboards</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <div className="text-red-400 font-bold mb-2">Error Tracking</div>
                <ul className="text-sm space-y-2">
                  <li>
                    <a href={`${GRAFANA_BASE}/d/logs-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-red-400 transition-colors flex items-center gap-1">
                      📝 Log Aggregation <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                  <li>
                    <a href={`${GRAFANA_BASE}/d/p1b-logs`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-red-400 transition-colors flex items-center gap-1">
                      📋 P1-B Logs <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <div className="text-yellow-400 font-bold mb-2">Service Logs</div>
                <ul className="text-sm space-y-2">
                  <li>
                    <a href={`${GRAFANA_BASE}/d/services-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-yellow-400 transition-colors flex items-center gap-1">
                      🔧 Service Status <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                  <li>
                    <a href={`${GRAFANA_BASE}/d/b3a96f3a-1ceb-480f-a7a7-c3b6f044dca7`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-yellow-400 transition-colors flex items-center gap-1">
                      🛡️ Safety Telemetry <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <div className="text-blue-400 font-bold mb-2">Trading Logs</div>
                <ul className="text-sm space-y-2">
                  <li>
                    <a href={`${GRAFANA_BASE}/d/b558bc08-73cf-4e97-80c2-6c0eb37aa564`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-blue-400 transition-colors flex items-center gap-1">
                      💹 Execution & Trading <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                  <li>
                    <a href={`${GRAFANA_BASE}/d/eff09fa9-6865-4a9e-b9d3-3568bb61b682`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-blue-400 transition-colors flex items-center gap-1">
                      🌾 Harvest Control <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                </ul>
              </div>
              <div>
                <div className="text-green-400 font-bold mb-2">AI & RL</div>
                <ul className="text-sm space-y-2">
                  <li>
                    <a href={`${GRAFANA_BASE}/d/rl-shadow-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-green-400 transition-colors flex items-center gap-1">
                      🧠 RL Shadow Monitor <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                  <li>
                    <a href={`${GRAFANA_BASE}/d/rl-shadow-performance`} target="_blank" rel="noopener noreferrer" className="text-gray-300 hover:text-green-400 transition-colors flex items-center gap-1">
                      🎯 RL Performance <span className="text-gray-600">↗</span>
                    </a>
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-blue-500/10 border border-blue-500/30 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <div className="text-2xl">ℹ️</div>
              <div>
                <div className="text-blue-400 font-semibold mb-1">Extended Time Range</div>
                <p className="text-gray-300 text-sm">
                  Log dashboard shows last 24 hours to ensure sufficient data visibility.
                  If panels appear empty, try expanding the time range to "Last 7 days" in the Grafana UI.
                </p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* ── All Dashboards Tab ── */}
      {activeTab === 'dashboards' && (
        <div className="space-y-6">
          {/* Dashboard Selector + Iframe */}
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <div className="flex items-center gap-4">
                <h2 className="text-2xl font-semibold text-white">Dashboard Viewer</h2>
                <select
                  value={selectedDashboard}
                  onChange={(e) => setSelectedDashboard(e.target.value)}
                  className="bg-gray-700 text-white border border-gray-600 rounded-lg px-4 py-2 text-sm focus:border-purple-500 focus:outline-none"
                >
                  {DASHBOARDS.map(d => (
                    <option key={d.uid} value={d.uid}>
                      {d.icon} {d.title}
                    </option>
                  ))}
                </select>
              </div>
              <a
                href={`${GRAFANA_BASE}/d/${selectedDashboard}`}
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg text-white text-sm font-medium transition-colors"
              >
                Full Screen ↗
              </a>
            </div>

            <div className="bg-gray-900 rounded-lg overflow-hidden shadow-2xl h-[800px]">
              <iframe
                src={`${GRAFANA_BASE}/d/${selectedDashboard}?kiosk=tv&refresh=30s&from=now-6h&to=now`}
                width="100%"
                height="100%"
                frameBorder="0"
                title={DASHBOARDS.find(d => d.uid === selectedDashboard)?.title ?? 'Dashboard'}
                className="w-full h-full"
              />
            </div>
          </div>

          {/* Dashboard Directory */}
          <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-6">📊 Dashboard Directory — {DASHBOARDS.length} Dashboards</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              {categories.map(cat => (
                <div key={cat}>
                  <div className="text-purple-400 font-bold mb-3 text-sm uppercase tracking-wider">{cat}</div>
                  <ul className="space-y-2">
                    {DASHBOARDS.filter(d => d.category === cat).map(d => (
                      <li key={d.uid}>
                        <a
                          href={`${GRAFANA_BASE}/d/${d.uid}`}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="text-gray-300 hover:text-purple-400 transition-colors flex items-center gap-2 text-sm group"
                        >
                          <span>{d.icon}</span>
                          <span className="group-hover:underline">{d.title}</span>
                          <span className="text-gray-600 text-xs">↗</span>
                        </a>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* ── Footer ── */}
      <footer className="border-t border-gray-700 pt-6 pb-4">
        <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-sm">
          <div>
            <div className="text-gray-500 font-semibold mb-2 uppercase text-xs tracking-wider">Infrastructure</div>
            <ul className="space-y-1">
              <li><a href={`${GRAFANA_BASE}/d/sys-overview-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">System Overview</a></li>
              <li><a href={`${GRAFANA_BASE}/d/infra-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">Infrastructure</a></li>
              <li><a href={`${GRAFANA_BASE}/d/services-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">Services</a></li>
              <li><a href={`${GRAFANA_BASE}/d/redis-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">Redis Metrics</a></li>
            </ul>
          </div>
          <div>
            <div className="text-gray-500 font-semibold mb-2 uppercase text-xs tracking-wider">Trading</div>
            <ul className="space-y-1">
              <li><a href={`${GRAFANA_BASE}/d/b558bc08-73cf-4e97-80c2-6c0eb37aa564`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">Execution & Trading</a></li>
              <li><a href={`${GRAFANA_BASE}/d/eff09fa9-6865-4a9e-b9d3-3568bb61b682`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">Harvest Control</a></li>
              <li><a href={`${GRAFANA_BASE}/d/1079a1b3-bc66-4d49-bafe-4a388f78f4de`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">Harvest Optimizer</a></li>
            </ul>
          </div>
          <div>
            <div className="text-gray-500 font-semibold mb-2 uppercase text-xs tracking-wider">AI & RL</div>
            <ul className="space-y-1">
              <li><a href={`${GRAFANA_BASE}/d/rl-shadow-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">RL Shadow Monitor</a></li>
              <li><a href={`${GRAFANA_BASE}/d/rl-shadow-performance`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">RL Performance</a></li>
              <li><a href={`${GRAFANA_BASE}/d/p1c-baseline`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">Performance Baseline</a></li>
            </ul>
          </div>
          <div>
            <div className="text-gray-500 font-semibold mb-2 uppercase text-xs tracking-wider">Logs & Safety</div>
            <ul className="space-y-1">
              <li><a href={`${GRAFANA_BASE}/d/logs-clean`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">Log Aggregation</a></li>
              <li><a href={`${GRAFANA_BASE}/d/p1b-logs`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">P1-B Logs</a></li>
              <li><a href={`${GRAFANA_BASE}/d/b3a96f3a-1ceb-480f-a7a7-c3b6f044dca7`} target="_blank" rel="noopener noreferrer" className="text-gray-400 hover:text-purple-400 transition-colors">Safety Telemetry</a></li>
            </ul>
          </div>
        </div>
        <div className="mt-4 pt-4 border-t border-gray-800 text-center">
          <a
            href={`${GRAFANA_BASE}/`}
            target="_blank"
            rel="noopener noreferrer"
            className="text-gray-500 hover:text-purple-400 text-xs transition-colors"
          >
            Open Grafana UI → {DASHBOARDS.length} dashboards available
          </a>
        </div>
      </footer>
    </div>
  );
}
