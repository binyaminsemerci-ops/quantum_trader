import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';
import { safeNum } from '../utils/formatters';

export default function System() {
  const [health, setHealth] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:8000/system/health')
      .then(res => res.json())
      .then(data => setHealth(data))
      .catch(err => console.error(err));
  }, []);

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white">System Monitoring</h1>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <InsightCard
          title="CPU Usage"
          value={`${safeNum(health?.cpu_usage, 1)}%`}
          icon="🖥️"
        />
        <InsightCard
          title="RAM Usage"
          value={`${safeNum(health?.ram_usage, 1)}%`}
          icon="💾"
        />
        <InsightCard
          title="Disk Usage"
          value={`${safeNum(health?.disk_usage, 1)}%`}
          icon="💿"
        />
        <InsightCard
          title="Uptime"
          value={`${health?.uptime_hours || 0}h`}
          icon="⏱️"
        />
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg p-6">
        <h2 className="text-xl font-semibold mb-4 text-white">Services Status</h2>
        <div className="grid grid-cols-2 gap-4">
          {health?.services && Object.entries(health.services).map(([service, status]: [string, any]) => (
            <div key={service} className="flex items-center justify-between p-4 bg-gray-800 rounded">
              <span className="text-gray-300 capitalize">{service}</span>
              <span className={`px-3 py-1 rounded text-xs font-semibold ${
                status === 'operational' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
              }`}>
                {status}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
