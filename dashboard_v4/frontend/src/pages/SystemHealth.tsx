import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface ContainerInfo {
  name: string;
  status: string;
  health: 'healthy' | 'unhealthy' | 'unknown';
  uptime: string;
}

interface SystemData {
  cpu_usage: number;
  ram_usage: number;
  disk_usage: number;
  containers_running: number;
  uptime_hours: number;
  docker_available_gb: number;
  docker_storage_note: string;
  storage_status: string;
  system_status: string;
  containers: ContainerInfo[];
}

export default function SystemHealth() {
  const [data, setData] = useState<SystemData | null>(null);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'overview' | 'metrics' | 'logs'>('overview');

  useEffect(() => {
    const fetchSystem = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/system/health`);
        if (!response.ok) throw new Error('Failed to fetch');
        const systemData = await response.json();
        
        // Parse container data
        const containers: ContainerInfo[] = Object.entries(systemData.containers || {}).map(([name, status]) => {
          const statusStr = status as string;
          const isHealthy = statusStr.toLowerCase().includes('healthy');
          const isUnhealthy = statusStr.toLowerCase().includes('unhealthy');
          
          return {
            name: name.replace('quantum_', ''),
            status: statusStr,
            health: isUnhealthy ? 'unhealthy' : isHealthy ? 'healthy' : 'unknown',
            uptime: statusStr.split('Up ')[1]?.split(' (')[0] || 'Unknown'
          };
        });

        setData({
          cpu_usage: systemData?.metrics?.cpu ?? 0,
          ram_usage: systemData?.metrics?.ram ?? 0,
          disk_usage: systemData?.metrics?.disk ?? 0,
          containers_running: systemData?.container_count ?? 0,
          uptime_hours: systemData?.metrics?.uptime_hours ?? 0,
          docker_available_gb: systemData?.metrics?.docker_available_gb ?? 0,
          docker_storage_note: systemData?.metrics?.docker_storage ?? 'N/A',
          storage_status: systemData?.metrics?.storage_status ?? '',
          system_status: systemData?.status ?? 'UNKNOWN',
          containers: containers.sort((a, b) => {
            // Sort by health status (unhealthy first), then by name
            if (a.health === 'unhealthy' && b.health !== 'unhealthy') return -1;
            if (a.health !== 'unhealthy' && b.health === 'unhealthy') return 1;
            return a.name.localeCompare(b.name);
          })
        });
        setLoading(false);
      } catch (err) {
        console.error('Failed to load system data:', err);
        setLoading(false);
      }
    };

    fetchSystem();
    const interval = setInterval(fetchSystem, 3000); // 3 second refresh for system metrics
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading system health...</div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-red-400">Failed to load system data</div>
      </div>
    );
  }

  const getHealthColor = (value: number, thresholds: { warning: number; critical: number }) => {
    if (value >= thresholds.critical) return 'red';
    if (value >= thresholds.warning) return 'yellow';
    return 'green';
  };

  const cpuColor = getHealthColor(data?.cpu_usage ?? 0, { warning: 70, critical: 85 });
  const ramColor = getHealthColor(data?.ram_usage ?? 0, { warning: 75, critical: 90 });
  const diskColor = getHealthColor(data?.disk_usage ?? 0, { warning: 80, critical: 95 });

  const healthyContainers = data.containers.filter(c => c.health === 'healthy').length;
  const unhealthyContainers = data.containers.filter(c => c.health === 'unhealthy').length;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-purple-400">System Health</h1>
        <div className="flex items-center gap-4">
          <div className={`px-4 py-2 rounded-lg font-bold ${
            data.system_status === 'HEALTHY' ? 'bg-green-500/20 text-green-400' :
            data.system_status === 'STRESSED' ? 'bg-yellow-500/20 text-yellow-400' :
            'bg-red-500/20 text-red-400'
          }`}>
            {data.system_status}
          </div>
          <div className="text-sm text-gray-400">
            Last updated: {new Date().toLocaleTimeString()}
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex space-x-2 bg-gray-800 rounded-lg p-1">
        <button
          onClick={() => setActiveTab('overview')}
          className={`px-6 py-2 rounded-lg font-medium transition-all ${
            activeTab === 'overview'
              ? 'bg-purple-500 text-white shadow-lg'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
        >
          📊 Overview
        </button>
        <button
          onClick={() => setActiveTab('metrics')}
          className={`px-6 py-2 rounded-lg font-medium transition-all ${
            activeTab === 'metrics'
              ? 'bg-purple-500 text-white shadow-lg'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
        >
          📈 Performance Metrics
        </button>
        <button
          onClick={() => setActiveTab('logs')}
          className={`px-6 py-2 rounded-lg font-medium transition-all ${
            activeTab === 'logs'
              ? 'bg-purple-500 text-white shadow-lg'
              : 'text-gray-400 hover:text-white hover:bg-gray-700'
          }`}
        >
          📝 Log Analysis
        </button>
      </div>

      {/* Overview Tab */}
      {activeTab === 'overview' && (
        <>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <InsightCard
          title="CPU Usage"
          value={`${data.cpu_usage.toFixed(1)}%`}
          subtitle="Processing load"
          color={cpuColor === 'green' ? 'text-green-400' : cpuColor === 'yellow' ? 'text-yellow-400' : 'text-red-400'}
        />
        
        <InsightCard
          title="RAM Usage"
          value={`${data.ram_usage.toFixed(1)}%`}
          subtitle="Memory consumption"
          color={ramColor === 'green' ? 'text-green-400' : ramColor === 'yellow' ? 'text-yellow-400' : 'text-red-400'}
        />
        
        <InsightCard
          title="Disk Usage"
          value={`${data.disk_usage.toFixed(1)}%`}
          subtitle="Storage capacity (OS)"
          color={diskColor === 'green' ? 'text-green-400' : diskColor === 'yellow' ? 'text-yellow-400' : 'text-red-400'}
        />
        
        <InsightCard
          title="Containers"
          value={data.containers_running.toString()}
          subtitle={`${healthyContainers} healthy, ${unhealthyContainers} issues`}
          color="text-blue-400"
        />
      </div>

      {/* Resource Gauges */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* CPU Gauge */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">CPU</h2>
          <div className="flex justify-center mb-4">
            <div className="relative w-32 h-32">
              <svg className="transform -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" fill="none" stroke="#374151" strokeWidth="8" />
                <circle
                  cx="50" cy="50" r="40" fill="none"
                  stroke={cpuColor === 'green' ? '#10b981' : cpuColor === 'yellow' ? '#f59e0b' : '#ef4444'}
                  strokeWidth="8"
                  strokeDasharray={`${data.cpu_usage * 2.51} 251`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold text-white">{data.cpu_usage.toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <div className="text-center text-sm text-gray-400">
            {data.cpu_usage < 70 ? 'Normal' : data.cpu_usage < 85 ? 'Warning' : 'Critical'}
          </div>
        </div>

        {/* RAM Gauge */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">RAM</h2>
          <div className="flex justify-center mb-4">
            <div className="relative w-32 h-32">
              <svg className="transform -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" fill="none" stroke="#374151" strokeWidth="8" />
                <circle
                  cx="50" cy="50" r="40" fill="none"
                  stroke={ramColor === 'green' ? '#10b981' : ramColor === 'yellow' ? '#f59e0b' : '#ef4444'}
                  strokeWidth="8"
                  strokeDasharray={`${data.ram_usage * 2.51} 251`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold text-white">{data.ram_usage.toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <div className="text-center text-sm text-gray-400">
            {data.ram_usage < 75 ? 'Normal' : data.ram_usage < 90 ? 'Warning' : 'Critical'}
          </div>
        </div>

        {/* Disk Gauge */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Disk (OS)</h2>
          <div className="flex justify-center mb-4">
            <div className="relative w-32 h-32">
              <svg className="transform -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" fill="none" stroke="#374151" strokeWidth="8" />
                <circle
                  cx="50" cy="50" r="40" fill="none"
                  stroke={diskColor === 'green' ? '#10b981' : diskColor === 'yellow' ? '#f59e0b' : '#ef4444'}
                  strokeWidth="8"
                  strokeDasharray={`${data.disk_usage * 2.51} 251`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold text-white">{data.disk_usage.toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <div className="text-center text-sm text-gray-400">
            {data.disk_usage < 80 ? 'Normal' : data.disk_usage < 95 ? 'Warning' : 'Critical'}
          </div>
        </div>
      </div>

      {/* Docker Volume Info */}
      <div className="bg-gradient-to-r from-green-500/20 to-blue-500/20 border border-green-500/30 rounded-lg p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-white mb-2">Docker Storage Volume</h2>
            <p className="text-gray-300">{data.docker_storage_note}</p>
            <p className="text-green-400 font-bold text-lg mt-2">{data.storage_status}</p>
          </div>
          <div className="text-right">
            <div className="text-4xl font-bold text-green-400">{data.docker_available_gb}GB</div>
            <div className="text-sm text-gray-400">Available Space</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left: System Stats */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">System Stats</h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Uptime:</span>
              <span className="text-white font-bold">{data.uptime_hours.toFixed(1)} hours</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Containers:</span>
              <span className="text-white font-bold">{data.containers_running}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Healthy Containers:</span>
              <span className="text-green-400 font-bold">{data.containers.filter(c => c.health === 'healthy').length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Unhealthy Containers:</span>
              <span className={`font-bold ${data.containers.filter(c => c.health === 'unhealthy').length > 0 ? 'text-red-400' : 'text-green-400'}`}>
                {data.containers.filter(c => c.health === 'unhealthy').length}
              </span>
            </div>
            <div className="flex justify-between border-t border-gray-700 pt-3">
              <span className="text-gray-400">System Status:</span>
              <span className={`font-bold ${
                data.system_status === 'HEALTHY' ? 'text-green-400' :
                data.system_status === 'STRESSED' ? 'text-yellow-400' :
                'text-red-400'
              }`}>
                {data.system_status}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Docker Volume:</span>
              <span className="text-green-400 font-bold">{data.docker_available_gb}GB free</span>
            </div>
          </div>
        </div>

        {/* Right: Quick Service Overview */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Quick Status Overview</h2>
          <div className="space-y-3">
            <div className="flex items-center justify-between bg-gray-700 rounded p-3">
              <span className="text-gray-300 font-semibold">CPU Load</span>
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  cpuColor === 'green' ? 'bg-green-500' :
                  cpuColor === 'yellow' ? 'bg-yellow-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm font-bold text-white">{data.cpu_usage.toFixed(1)}%</span>
              </div>
            </div>
            <div className="flex items-center justify-between bg-gray-700 rounded p-3">
              <span className="text-gray-300 font-semibold">Memory</span>
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  ramColor === 'green' ? 'bg-green-500' :
                  ramColor === 'yellow' ? 'bg-yellow-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm font-bold text-white">{data.ram_usage.toFixed(1)}%</span>
              </div>
            </div>
            <div className="flex items-center justify-between bg-gray-700 rounded p-3">
              <span className="text-gray-300 font-semibold">Storage (OS)</span>
              <div className="flex items-center gap-2">
                <div className={`w-3 h-3 rounded-full ${
                  diskColor === 'green' ? 'bg-green-500' :
                  diskColor === 'yellow' ? 'bg-yellow-500' : 'bg-red-500'
                }`}></div>
                <span className="text-sm font-bold text-white">{data.disk_usage.toFixed(1)}%</span>
              </div>
            </div>
            <div className="flex items-center justify-between bg-gray-700 rounded p-3">
              <span className="text-gray-300 font-semibold">Docker Volume</span>
              <div className="flex items-center gap-2">
                <div className="w-3 h-3 rounded-full bg-green-500"></div>
                <span className="text-sm font-bold text-green-400">{data.docker_available_gb}GB</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* All Containers Status Table */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold text-white mb-4">
          Container Status ({data.containers_running} Total)
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-700">
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Service Name</th>
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Uptime</th>
                <th className="text-center py-3 px-4 text-gray-400 font-semibold">Health Status</th>
                <th className="text-left py-3 px-4 text-gray-400 font-semibold">Full Status</th>
              </tr>
            </thead>
            <tbody>
              {data.containers.map((container) => (
                <tr key={container.name} className="border-b border-gray-700 hover:bg-gray-750 transition-colors">
                  <td className="py-3 px-4 text-white font-medium">
                    {container.name.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')}
                  </td>
                  <td className="py-3 px-4 text-gray-300">{container.uptime}</td>
                  <td className="py-3 px-4 text-center">
                    <span className={`px-3 py-1 rounded-full text-xs font-semibold inline-flex items-center gap-2 ${
                      container.health === 'healthy' ? 'bg-green-500/20 text-green-400' :
                      container.health === 'unhealthy' ? 'bg-red-500/20 text-red-400' :
                      'bg-gray-500/20 text-gray-400'
                    }`}>
                      <div className={`w-2 h-2 rounded-full ${
                        container.health === 'healthy' ? 'bg-green-500' :
                        container.health === 'unhealthy' ? 'bg-red-500' : 'bg-gray-500'
                      }`}></div>
                      {container.health.charAt(0).toUpperCase() + container.health.slice(1)}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-gray-400 text-xs">{container.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
        </>
      )}

      {/* Performance Metrics Tab - Grafana P1-C Dashboard */}
      {activeTab === 'metrics' && (
        <div className="space-y-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex justify-between items-center mb-4">
              <div>
                <h2 className="text-2xl font-semibold text-white">Performance Baseline (P1-C)</h2>
                <p className="text-gray-400 mt-1">Real-time system metrics and resource monitoring</p>
              </div>
              <a
                href="/grafana/d/p1c-baseline"
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-purple-500 hover:bg-purple-600 rounded-lg text-white font-medium transition-colors"
              >
                Open in Grafana ↗
              </a>
            </div>
            
            <div className="bg-gray-900 rounded-lg overflow-hidden" style={{ height: '800px' }}>
              <iframe
                src="/grafana/d/p1c-baseline?kiosk=tv&refresh=30s&from=now-6h&to=now"
                width="100%"
                height="100%"
                frameBorder="0"
                title="P1-C Performance Baseline"
                className="w-full h-full"
              />
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="bg-gray-700 rounded p-3">
                <div className="text-gray-400">Dashboard</div>
                <div className="text-white font-semibold">P1-C Performance Baseline</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-gray-400">Refresh Rate</div>
                <div className="text-green-400 font-semibold">30 seconds</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-gray-400">Time Range</div>
                <div className="text-blue-400 font-semibold">Last 6 hours</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-500/20 to-blue-500/20 border border-purple-500/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-3">📊 Included Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-purple-400 font-bold">System Resources</div>
                <ul className="text-gray-300 text-sm mt-2 space-y-1">
                  <li>• CPU Usage</li>
                  <li>• Memory Usage</li>
                  <li>• Disk Usage</li>
                </ul>
              </div>
              <div>
                <div className="text-blue-400 font-bold">Containers</div>
                <ul className="text-gray-300 text-sm mt-2 space-y-1">
                  <li>• Running Count</li>
                  <li>• CPU per Container</li>
                  <li>• Memory per Container</li>
                </ul>
              </div>
              <div>
                <div className="text-green-400 font-bold">Redis</div>
                <ul className="text-gray-300 text-sm mt-2 space-y-1">
                  <li>• Operations/sec</li>
                  <li>• Connected Clients</li>
                  <li>• Memory Usage</li>
                </ul>
              </div>
              <div>
                <div className="text-yellow-400 font-bold">Network</div>
                <ul className="text-gray-300 text-sm mt-2 space-y-1">
                  <li>• Traffic I/O</li>
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
                <h2 className="text-2xl font-semibold text-white">Log Analysis (P1-B)</h2>
                <p className="text-gray-400 mt-1">Centralized log aggregation and error tracking</p>
              </div>
              <a
                href="/grafana/d/p1b-logs"
                target="_blank"
                rel="noopener noreferrer"
                className="px-4 py-2 bg-purple-500 hover:bg-purple-600 rounded-lg text-white font-medium transition-colors"
              >
                Open in Grafana ↗
              </a>
            </div>
            
            <div className="bg-gray-900 rounded-lg overflow-hidden" style={{ height: '800px' }}>
              <iframe
                src="/grafana/d/p1b-logs?kiosk=tv&refresh=30s&from=now-1h&to=now"
                width="100%"
                height="100%"
                frameBorder="0"
                title="P1-B Log Analysis"
                className="w-full h-full"
              />
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
              <div className="bg-gray-700 rounded p-3">
                <div className="text-gray-400">Dashboard</div>
                <div className="text-white font-semibold">P1-B Log Analysis</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-gray-400">Refresh Rate</div>
                <div className="text-green-400 font-semibold">30 seconds</div>
              </div>
              <div className="bg-gray-700 rounded p-3">
                <div className="text-gray-400">Time Range</div>
                <div className="text-blue-400 font-semibold">Last 1 hour</div>
              </div>
            </div>
          </div>

          <div className="bg-gradient-to-r from-purple-500/20 to-red-500/20 border border-purple-500/30 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-3">📝 Log Features</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-red-400 font-bold">Error Tracking</div>
                <ul className="text-gray-300 text-sm mt-2 space-y-1">
                  <li>• Error Rates</li>
                  <li>• Critical Logs</li>
                  <li>• Exception Types</li>
                </ul>
              </div>
              <div>
                <div className="text-yellow-400 font-bold">Service Logs</div>
                <ul className="text-gray-300 text-sm mt-2 space-y-1">
                  <li>• AI Engine</li>
                  <li>• Auto Executor</li>
                  <li>• All Containers</li>
                </ul>
              </div>
              <div>
                <div className="text-blue-400 font-bold">Search</div>
                <ul className="text-gray-300 text-sm mt-2 space-y-1">
                  <li>• LogQL Queries</li>
                  <li>• Pattern Matching</li>
                  <li>• Time Filtering</li>
                </ul>
              </div>
              <div>
                <div className="text-green-400 font-bold">Insights</div>
                <ul className="text-gray-300 text-sm mt-2 space-y-1">
                  <li>• Log Volume</li>
                  <li>• Top Errors</li>
                  <li>• Trend Analysis</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
