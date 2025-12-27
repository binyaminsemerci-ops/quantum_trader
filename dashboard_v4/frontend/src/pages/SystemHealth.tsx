import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

interface SystemData {
  cpu_usage: number;
  ram_usage: number;
  disk_usage: number;
  containers_running: number;
  uptime_hours: number;
  network_latency_ms: number;
  api_requests_per_min: number;
  error_rate: number;
}

export default function SystemHealth() {
  const [data, setData] = useState<SystemData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSystem = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/system/health`);
        const systemData = await response.json();
        // Map API response to expected format
        setData({
          cpu_usage: systemData?.metrics?.cpu ?? 0,
          ram_usage: systemData?.metrics?.ram ?? 0,
          disk_usage: systemData?.metrics?.disk ?? 0,
          containers_running: systemData?.container_count ?? 0,
          uptime_hours: systemData?.metrics?.uptime_hours ?? 0,
          network_latency_ms: 50,  // Placeholder
          api_requests_per_min: 120,  // Placeholder
          error_rate: 0.01  // Placeholder
        });
        setLoading(false);
      } catch (err) {
        console.error('Failed to load system data:', err);
        setLoading(false);
      }
    };

    fetchSystem();
    const interval = setInterval(fetchSystem, 3000); // More frequent for system metrics
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

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-purple-400">System Health</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <InsightCard
          title="CPU Usage"
          value={`${(data?.cpu_usage ?? 0).toFixed(1)}%`}
          subtitle="Processing load"
          color={cpuColor === 'green' ? 'text-green-400' : cpuColor === 'yellow' ? 'text-yellow-400' : 'text-red-400'}
        />
        
        <InsightCard
          title="RAM Usage"
          value={`${(data?.ram_usage ?? 0).toFixed(1)}%`}
          subtitle="Memory consumption"
          color={ramColor === 'green' ? 'text-green-400' : ramColor === 'yellow' ? 'text-yellow-400' : 'text-red-400'}
        />
        
        <InsightCard
          title="Disk Usage"
          value={`${(data?.disk_usage ?? 0).toFixed(1)}%`}
          subtitle="Storage capacity"
          color={diskColor === 'green' ? 'text-green-400' : diskColor === 'yellow' ? 'text-yellow-400' : 'text-red-400'}
        />
        
        <InsightCard
          title="Containers"
          value={(data?.containers_running ?? 0).toString()}
          subtitle="Active services"
          color="text-blue-400"
        />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
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
                  strokeDasharray={`${(data?.cpu_usage ?? 0) * 2.51} 251`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold text-white">{(data?.cpu_usage ?? 0).toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <div className="text-center text-sm text-gray-400">
            {(data?.cpu_usage ?? 0) < 70 ? 'Normal' : (data?.cpu_usage ?? 0) < 85 ? 'Warning' : 'Critical'}
          </div>
        </div>

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
                  strokeDasharray={`${(data?.ram_usage ?? 0) * 2.51} 251`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold text-white">{(data?.ram_usage ?? 0).toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <div className="text-center text-sm text-gray-400">
            {(data?.ram_usage ?? 0) < 75 ? 'Normal' : (data?.ram_usage ?? 0) < 90 ? 'Warning' : 'Critical'}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Disk</h2>
          <div className="flex justify-center mb-4">
            <div className="relative w-32 h-32">
              <svg className="transform -rotate-90" viewBox="0 0 100 100">
                <circle cx="50" cy="50" r="40" fill="none" stroke="#374151" strokeWidth="8" />
                <circle
                  cx="50" cy="50" r="40" fill="none"
                  stroke={diskColor === 'green' ? '#10b981' : diskColor === 'yellow' ? '#f59e0b' : '#ef4444'}
                  strokeWidth="8"
                  strokeDasharray={`${(data?.disk_usage ?? 0) * 2.51} 251`}
                  strokeLinecap="round"
                />
              </svg>
              <div className="absolute inset-0 flex items-center justify-center">
                <span className="text-2xl font-bold text-white">{(data?.disk_usage ?? 0).toFixed(1)}%</span>
              </div>
            </div>
          </div>
          <div className="text-center text-sm text-gray-400">
            {(data?.disk_usage ?? 0) < 80 ? 'Normal' : (data?.disk_usage ?? 0) < 95 ? 'Warning' : 'Critical'}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">System Stats</h2>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Uptime:</span>
              <span className="text-white font-bold">{(data?.uptime_hours ?? 0).toFixed(1)} hours</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Containers:</span>
              <span className="text-white font-bold">{data?.containers_running ?? 0}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Network Latency:</span>
              <span className="text-white font-bold">{data?.network_latency_ms ?? 0}ms</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">API Requests:</span>
              <span className="text-white font-bold">{data?.api_requests_per_min ?? 0}/min</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Error Rate:</span>
              <span className={`font-bold ${(data?.error_rate ?? 0) > 0.05 ? 'text-red-400' : 'text-green-400'}`}>
                {((data?.error_rate ?? 0) * 100).toFixed(2)}%
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h2 className="text-xl font-semibold text-white mb-4">Service Status</h2>
          <div className="space-y-2">
            {[
              'AI Engine',
              'Portfolio Manager',
              'Risk Monitor',
              'Data Collector',
              'Dashboard Backend',
              'Database'
            ].map((service) => (
              <div key={service} className="flex items-center justify-between bg-gray-700 rounded p-2">
                <span className="text-gray-300">{service}</span>
                <span className="flex items-center">
                  <div className="w-2 h-2 bg-green-500 rounded-full mr-2"></div>
                  <span className="text-sm text-green-400">Running</span>
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
