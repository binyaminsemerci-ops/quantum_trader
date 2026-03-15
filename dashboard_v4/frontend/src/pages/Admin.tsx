import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = '/api';

interface Service {
  name: string;
  load: string;
  active: string;
  sub: string;
  description: string;
}

interface SystemInfo {
  cpu_percent: number;
  cpu_count: number;
  ram_total_gb: number;
  ram_used_gb: number;
  ram_percent: number;
  disk_total_gb: number;
  disk_used_gb: number;
  disk_percent: number;
  load_avg: number[] | null;
}

interface UserInfo {
  username: string;
  role: string;
}

export default function Admin({ token }: { token: string | null }) {
  const [services, setServices] = useState<Service[]>([]);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [users, setUsers] = useState<UserInfo[]>([]);
  const [loading, setLoading] = useState(true);
  const [logs, setLogs] = useState<{ service: string; lines: string[] } | null>(null);
  const [restartStatus, setRestartStatus] = useState<Record<string, string>>({});

  const headers: Record<string, string> = {};
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const fetchData = async () => {
    if (!token) { setLoading(false); return; }
    try {
      const [svcRes, sysRes, usrRes] = await Promise.all([
        fetch(`${API_BASE_URL}/admin/services`, { headers }),
        fetch(`${API_BASE_URL}/admin/system`, { headers }),
        fetch(`${API_BASE_URL}/admin/users`, { headers }),
      ]);
      if (svcRes.ok) { const d = await svcRes.json(); setServices(d.services || []); }
      if (sysRes.ok) setSystem(await sysRes.json());
      if (usrRes.ok) setUsers(await usrRes.json());
    } catch (err) {
      console.error('Failed to load admin data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [token]);

  const restartService = async (name: string) => {
    if (!token) return;
    setRestartStatus((s) => ({ ...s, [name]: 'restarting...' }));
    try {
      const res = await fetch(`${API_BASE_URL}/admin/services/${name}/restart`, {
        method: 'POST', headers,
      });
      const data = await res.json();
      setRestartStatus((s) => ({ ...s, [name]: data.status }));
      setTimeout(() => { setRestartStatus((s) => { const c = { ...s }; delete c[name]; return c; }); fetchData(); }, 3000);
    } catch { setRestartStatus((s) => ({ ...s, [name]: 'error' })); }
  };

  const viewLogs = async (name: string) => {
    if (!token) return;
    if (logs?.service === name) { setLogs(null); return; }
    try {
      const res = await fetch(`${API_BASE_URL}/admin/services/${name}/logs?lines=50`, { headers });
      if (res.ok) { const data = await res.json(); setLogs({ service: name, lines: data.lines || [] }); }
    } catch { /* ignore */ }
  };

  if (!token) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Admin access required. Please sign in with an admin account.</div>
      </div>
    );
  }

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="text-gray-400">Loading admin panel...</div></div>;
  }

  const runningCount = services.filter((s) => s.active === 'active').length;
  const failedCount = services.filter((s) => s.active === 'failed').length;

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-green-400">Admin Dashboard</h1>

      {/* System overview */}
      {system && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
          <InsightCard title="CPU" value={`${system.cpu_percent}%`}
            subtitle={`${system.cpu_count} cores`}
            color={system.cpu_percent > 80 ? 'text-red-400' : 'text-green-400'} />
          <InsightCard title="RAM" value={`${system.ram_percent}%`}
            subtitle={`${system.ram_used_gb}/${system.ram_total_gb} GB`}
            color={system.ram_percent > 85 ? 'text-red-400' : 'text-green-400'} />
          <InsightCard title="Disk" value={`${system.disk_percent}%`}
            subtitle={`${system.disk_used_gb}/${system.disk_total_gb} GB`}
            color={system.disk_percent > 90 ? 'text-red-400' : 'text-green-400'} />
          <InsightCard title="Services" value={`${runningCount}/${services.length}`}
            subtitle={failedCount > 0 ? `${failedCount} failed` : 'All healthy'}
            color={failedCount > 0 ? 'text-red-400' : 'text-green-400'} />
          <InsightCard title="Load Avg" value={system.load_avg ? system.load_avg[0].toFixed(2) : 'N/A'}
            subtitle={system.load_avg ? `${system.load_avg[1].toFixed(2)} / ${system.load_avg[2].toFixed(2)}` : undefined} />
        </div>
      )}

      {/* Users */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 p-4">
        <h2 className="text-lg font-semibold text-white mb-3">Users</h2>
        <div className="flex gap-3">
          {users.map((u) => (
            <div key={u.username} className="px-4 py-2 bg-gray-700 rounded-lg">
              <span className="text-white font-medium">{u.username}</span>
              <span className={`ml-2 px-2 py-0.5 rounded text-xs ${
                u.role === 'admin' ? 'bg-red-900/50 text-red-400' :
                u.role === 'analyst' ? 'bg-blue-900/50 text-blue-400' :
                'bg-gray-600 text-gray-300'
              }`}>{u.role}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Services list */}
      <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
        <div className="px-4 py-3 border-b border-gray-700 flex items-center justify-between">
          <h2 className="font-semibold text-white">Services ({services.length})</h2>
          <button onClick={fetchData} className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded text-sm transition">Refresh</button>
        </div>
        <div className="max-h-96 overflow-y-auto">
          <table className="w-full text-sm">
            <thead className="sticky top-0 bg-gray-800">
              <tr className="border-b border-gray-700">
                <th className="px-4 py-2 text-left text-gray-400">Service</th>
                <th className="px-4 py-2 text-center text-gray-400">Status</th>
                <th className="px-4 py-2 text-left text-gray-400">Description</th>
                <th className="px-4 py-2 text-right text-gray-400">Actions</th>
              </tr>
            </thead>
            <tbody>
              {services.map((svc) => (
                <tr key={svc.name} className="border-b border-gray-700/30 hover:bg-gray-750">
                  <td className="px-4 py-2 text-white font-mono text-xs">{svc.name}</td>
                  <td className="px-4 py-2 text-center">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      svc.sub === 'running' ? 'bg-green-900/50 text-green-400' :
                      svc.active === 'failed' ? 'bg-red-900/50 text-red-400' :
                      'bg-gray-700 text-gray-400'
                    }`}>{svc.sub}</span>
                  </td>
                  <td className="px-4 py-2 text-gray-400 text-xs">{svc.description}</td>
                  <td className="px-4 py-2 text-right">
                    <div className="flex items-center justify-end gap-1">
                      {restartStatus[svc.name] && (
                        <span className="text-yellow-400 text-xs mr-1">{restartStatus[svc.name]}</span>
                      )}
                      <button onClick={() => viewLogs(svc.name)}
                        className="px-2 py-1 bg-gray-700 hover:bg-gray-600 text-gray-300 rounded text-xs transition">
                        {logs?.service === svc.name ? 'Hide' : 'Logs'}
                      </button>
                      <button onClick={() => restartService(svc.name)}
                        className="px-2 py-1 bg-orange-600/20 text-orange-400 rounded text-xs hover:bg-orange-600/40 transition">
                        Restart
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Logs viewer */}
      {logs && (
        <div className="bg-gray-800 rounded-xl border border-gray-700 overflow-hidden">
          <div className="px-4 py-3 border-b border-gray-700">
            <h3 className="font-medium text-white">Logs: {logs.service}</h3>
          </div>
          <pre className="p-4 text-xs text-gray-300 overflow-x-auto max-h-64 overflow-y-auto font-mono">
            {logs.lines.join('\n')}
          </pre>
        </div>
      )}
    </div>
  );
}
