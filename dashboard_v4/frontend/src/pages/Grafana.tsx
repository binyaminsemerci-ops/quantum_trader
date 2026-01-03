import { useState } from 'react';

export default function Grafana() {
  const [activeTab, setActiveTab] = useState<'metrics' | 'logs'>('metrics');

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-purple-400">Grafana Monitoring</h1>
          <p className="text-gray-400 mt-2">Real-time metrics and log analysis dashboards</p>
        </div>
        <a
          href="/grafana/"
          target="_blank"
          rel="noopener noreferrer"
          className="px-4 py-2 bg-purple-500 hover:bg-purple-600 rounded-lg text-white font-medium transition-colors"
        >
          Open Full Grafana ↗
        </a>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-2 bg-gray-800 rounded-lg p-1">
        <button
          onClick={() => setActiveTab('metrics')}
          className={`flex-1 px-6 py-3 rounded-lg font-medium transition-all ${
            activeTab === 'metrics'
              ? 'bg-purple-500 text-white shadow-lg'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
        >
          📈 Performance Metrics
        </button>
        <button
          onClick={() => setActiveTab('logs')}
          className={`flex-1 px-6 py-3 rounded-lg font-medium transition-all ${
            activeTab === 'logs'
              ? 'bg-purple-500 text-white shadow-lg'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
        >
          📝 Log Analysis
        </button>
      </div>

      {/* Performance Metrics Tab - Grafana P1-C Dashboard */}
      {activeTab === 'metrics' && (
        <div className="space-y-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h2 className="text-2xl font-semibold text-white">P1-C: Performance Baseline</h2>
                <p className="text-gray-400 mt-1">System resources, containers, Redis, and network metrics</p>
              </div>
              <a
                href="/grafana/d/p1c-baseline"
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg text-white text-sm font-medium transition-colors"
              >
                Full Screen ↗
              </a>
            </div>
            
            <div className="bg-gray-900 rounded-lg overflow-hidden shadow-2xl" style={{ height: '900px' }}>
              <iframe
                src="/grafana/d/p1c-baseline?kiosk=tv&refresh=30s&from=now-6h&to=now"
                width="100%"
                height="100%"
                frameBorder="0"
                title="P1-C Performance Baseline"
                className="w-full h-full"
              />
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-gray-400 mb-1">Dashboard</div>
                <div className="text-white font-semibold">P1-C Performance Baseline</div>
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
                <div className="text-gray-400 mb-1">Total Panels</div>
                <div className="text-purple-400 font-semibold">11 Metrics</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">📊 Included Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <div className="text-purple-400 font-bold mb-2">System Resources</div>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• CPU Usage (%)</li>
                  <li>• Memory Usage (%)</li>
                  <li>• Disk Usage (%)</li>
                </ul>
              </div>
              <div>
                <div className="text-blue-400 font-bold mb-2">Containers</div>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• Running Count</li>
                  <li>• CPU per Container</li>
                  <li>• Memory per Container</li>
                </ul>
              </div>
              <div>
                <div className="text-green-400 font-bold mb-2">Redis</div>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• Operations/sec</li>
                  <li>• Connected Clients</li>
                  <li>• Memory Usage</li>
                </ul>
              </div>
              <div>
                <div className="text-yellow-400 font-bold mb-2">Network & Storage</div>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• Network Traffic I/O</li>
                  <li>• Prometheus Metrics</li>
                  <li>• Storage Usage</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Log Analysis Tab - Grafana P1-B Dashboard */}
      {activeTab === 'logs' && (
        <div className="space-y-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h2 className="text-2xl font-semibold text-white">P1-B: Log Aggregation</h2>
                <p className="text-gray-400 mt-1">Centralized log collection with error tracking and analysis</p>
              </div>
              <a
                href="/grafana/d/p1b-logs"
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-blue-500 hover:bg-blue-600 rounded-lg text-white text-sm font-medium transition-colors"
              >
                Full Screen ↗
              </a>
            </div>
            
            <div className="bg-gray-900 rounded-lg overflow-hidden shadow-2xl" style={{ height: '900px' }}>
              <iframe
                src="/grafana/d/p1b-logs?kiosk=tv&refresh=30s&from=now-24h&to=now"
                width="100%"
                height="100%"
                frameBorder="0"
                title="P1-B Log Analysis"
                className="w-full h-full"
              />
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="text-gray-400 mb-1">Dashboard</div>
                <div className="text-white font-semibold">P1-B Log Aggregation</div>
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

          <div className="bg-gradient-to-r from-purple-500/20 to-red-500/20 border border-purple-500/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">📝 Log Analysis Features</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6">
              <div>
                <div className="text-red-400 font-bold mb-2">Error Tracking</div>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• Error Rates by Level</li>
                  <li>• Critical Logs</li>
                  <li>• Exception Types</li>
                </ul>
              </div>
              <div>
                <div className="text-yellow-400 font-bold mb-2">Service Logs</div>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• AI Engine Logs</li>
                  <li>• Auto Executor Logs</li>
                  <li>• All Container Logs</li>
                </ul>
              </div>
              <div>
                <div className="text-blue-400 font-bold mb-2">Search & Filter</div>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• LogQL Queries</li>
                  <li>• Pattern Matching</li>
                  <li>• Time Filtering</li>
                </ul>
              </div>
              <div>
                <div className="text-green-400 font-bold mb-2">Insights</div>
                <ul className="text-gray-300 text-sm space-y-1">
                  <li>• Log Volume Trends</li>
                  <li>• Top Error Messages</li>
                  <li>• Order Flow Timeline</li>
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
    </div>
  );
}
