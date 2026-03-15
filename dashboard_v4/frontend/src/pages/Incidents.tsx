import { useEffect, useState } from 'react';
import InsightCard from '../components/InsightCard';

const API_BASE_URL = '/api';

interface Incident {
  id: number;
  title: string;
  description: string | null;
  severity: string;
  status: string;
  category: string | null;
  affected_services: string | null;
  root_cause: string | null;
  resolution: string | null;
  reported_by: string;
  assigned_to: string | null;
  opened_at: string;
  resolved_at: string | null;
  closed_at: string | null;
}

interface IncidentStats {
  total: number;
  open: number;
  resolved: number;
  by_severity: Record<string, number>;
  by_category: Record<string, number>;
}

const SEVERITY_COLORS: Record<string, string> = {
  critical: 'bg-red-900/50 text-red-400',
  high: 'bg-orange-900/50 text-orange-400',
  medium: 'bg-yellow-900/50 text-yellow-400',
  low: 'bg-blue-900/50 text-blue-400',
};

const STATUS_COLORS: Record<string, string> = {
  open: 'bg-red-900/50 text-red-400',
  investigating: 'bg-yellow-900/50 text-yellow-400',
  resolved: 'bg-green-900/50 text-green-400',
  closed: 'bg-gray-700 text-gray-400',
};

export default function Incidents({ token }: { token: string | null }) {
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [stats, setStats] = useState<IncidentStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterStatus, setFilterStatus] = useState('');
  const [filterSeverity, setFilterSeverity] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [selectedIncident, setSelectedIncident] = useState<Incident | null>(null);

  const [form, setForm] = useState({
    title: '', description: '', severity: 'medium', category: '', affected_services: '',
  });

  const headers: Record<string, string> = { 'Content-Type': 'application/json' };
  if (token) headers['Authorization'] = `Bearer ${token}`;

  const fetchData = async () => {
    try {
      const params = new URLSearchParams();
      if (filterStatus) params.set('status', filterStatus);
      if (filterSeverity) params.set('severity', filterSeverity);
      const qs = params.toString() ? `?${params}` : '';
      const [incRes, statsRes] = await Promise.all([
        fetch(`${API_BASE_URL}/incidents/${qs}`),
        fetch(`${API_BASE_URL}/incidents/stats`),
      ]);
      if (incRes.ok) setIncidents(await incRes.json());
      if (statsRes.ok) setStats(await statsRes.json());
    } catch (err) {
      console.error('Failed to load incidents:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { fetchData(); }, [filterStatus, filterSeverity]);

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!token) return alert('Login required');
    try {
      const res = await fetch(`${API_BASE_URL}/incidents/`, {
        method: 'POST', headers, body: JSON.stringify({
          title: form.title,
          description: form.description || null,
          severity: form.severity,
          category: form.category || null,
          affected_services: form.affected_services || null,
        }),
      });
      if (res.ok) {
        setShowForm(false);
        setForm({ title: '', description: '', severity: 'medium', category: '', affected_services: '' });
        fetchData();
      }
    } catch (err) { console.error('Create failed:', err); }
  };

  const updateStatus = async (id: number, newStatus: string) => {
    if (!token) return alert('Login required');
    try {
      const res = await fetch(`${API_BASE_URL}/incidents/${id}`, {
        method: 'PUT', headers, body: JSON.stringify({ status: newStatus }),
      });
      if (res.ok) fetchData();
    } catch (err) { console.error('Update failed:', err); }
  };

  if (loading) {
    return <div className="flex items-center justify-center h-64"><div className="text-gray-400">Loading incidents...</div></div>;
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-green-400">Incident Tracking</h1>
        <div className="flex gap-3">
          <select title="Filter by status" value={filterStatus} onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm">
            <option value="">All Status</option>
            <option value="open">Open</option>
            <option value="investigating">Investigating</option>
            <option value="resolved">Resolved</option>
            <option value="closed">Closed</option>
          </select>
          <select title="Filter by severity" value={filterSeverity} onChange={(e) => setFilterSeverity(e.target.value)}
            className="px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white text-sm">
            <option value="">All Severity</option>
            <option value="critical">Critical</option>
            <option value="high">High</option>
            <option value="medium">Medium</option>
            <option value="low">Low</option>
          </select>
          {token && (
            <button onClick={() => setShowForm(!showForm)}
              className="px-4 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-medium transition">
              {showForm ? 'Cancel' : '+ Report Incident'}
            </button>
          )}
        </div>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <InsightCard title="Total Incidents" value={stats.total} />
          <InsightCard title="Open" value={stats.open} color={stats.open > 0 ? 'text-red-400' : 'text-green-400'} />
          <InsightCard title="Resolved" value={stats.resolved} color="text-green-400" />
          <InsightCard title="Critical" value={stats.by_severity['critical'] ?? 0}
            color={(stats.by_severity['critical'] ?? 0) > 0 ? 'text-red-400' : 'text-gray-400'} />
        </div>
      )}

      {/* Report form */}
      {showForm && (
        <form onSubmit={handleCreate} className="bg-gray-800 p-6 rounded-xl border border-gray-700 space-y-4">
          <h2 className="text-lg font-semibold text-red-400">Report New Incident</h2>
          <input placeholder="Title *" required value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm" />
          <textarea placeholder="Description" value={form.description} onChange={(e) => setForm({ ...form, description: e.target.value })}
            className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm h-20" />
          <div className="grid grid-cols-3 gap-3">
            <select title="Severity" value={form.severity} onChange={(e) => setForm({ ...form, severity: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm">
              <option value="low">Low</option><option value="medium">Medium</option>
              <option value="high">High</option><option value="critical">Critical</option>
            </select>
            <input placeholder="Category (e.g. system, trading)" value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm" />
            <input placeholder="Affected services" value={form.affected_services} onChange={(e) => setForm({ ...form, affected_services: e.target.value })}
              className="px-3 py-2 bg-gray-700 border border-gray-600 rounded text-white text-sm" />
          </div>
          <button type="submit" className="px-6 py-2 bg-red-600 hover:bg-red-500 text-white rounded-lg text-sm font-medium transition">
            Report Incident
          </button>
        </form>
      )}

      {/* Incident list */}
      <div className="space-y-3">
        {incidents.length === 0 ? (
          <div className="bg-gray-800 rounded-xl border border-gray-700 p-8 text-center text-gray-500">No incidents found</div>
        ) : incidents.map((inc) => (
          <div key={inc.id} className="bg-gray-800 rounded-xl border border-gray-700 p-4 hover:border-gray-600 transition">
            <div className="flex items-start justify-between">
              <div className="flex-1">
                <div className="flex items-center gap-2 mb-1">
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${SEVERITY_COLORS[inc.severity] || 'bg-gray-700 text-gray-400'}`}>
                    {inc.severity.toUpperCase()}
                  </span>
                  <span className={`px-2 py-0.5 rounded text-xs font-medium ${STATUS_COLORS[inc.status] || 'bg-gray-700 text-gray-400'}`}>
                    {inc.status}
                  </span>
                  {inc.category && <span className="text-xs text-gray-500">{inc.category}</span>}
                  <span className="text-xs text-gray-600">#{inc.id}</span>
                </div>
                <h3 className="text-white font-medium cursor-pointer" onClick={() => setSelectedIncident(selectedIncident?.id === inc.id ? null : inc)}>
                  {inc.title}
                </h3>
                <p className="text-sm text-gray-400 mt-1">
                  {inc.reported_by} — {new Date(inc.opened_at).toLocaleString()}
                  {inc.affected_services && <span> — Services: {inc.affected_services}</span>}
                </p>
              </div>
              {token && inc.status !== 'closed' && (
                <div className="flex gap-1 ml-4">
                  {inc.status === 'open' && (
                    <button onClick={() => updateStatus(inc.id, 'investigating')} className="px-2 py-1 bg-yellow-600/20 text-yellow-400 rounded text-xs hover:bg-yellow-600/40 transition">Investigate</button>
                  )}
                  {(inc.status === 'open' || inc.status === 'investigating') && (
                    <button onClick={() => updateStatus(inc.id, 'resolved')} className="px-2 py-1 bg-green-600/20 text-green-400 rounded text-xs hover:bg-green-600/40 transition">Resolve</button>
                  )}
                  {inc.status === 'resolved' && (
                    <button onClick={() => updateStatus(inc.id, 'closed')} className="px-2 py-1 bg-gray-600/20 text-gray-400 rounded text-xs hover:bg-gray-600/40 transition">Close</button>
                  )}
                </div>
              )}
            </div>
            {selectedIncident?.id === inc.id && (
              <div className="mt-4 pt-4 border-t border-gray-700 space-y-2 text-sm">
                {inc.description && <div><span className="text-gray-500">Description:</span><p className="text-gray-300">{inc.description}</p></div>}
                {inc.root_cause && <div><span className="text-gray-500">Root Cause:</span><p className="text-orange-300">{inc.root_cause}</p></div>}
                {inc.resolution && <div><span className="text-gray-500">Resolution:</span><p className="text-green-300">{inc.resolution}</p></div>}
                {inc.resolved_at && <div className="text-gray-500">Resolved: {new Date(inc.resolved_at).toLocaleString()}</div>}
                {inc.closed_at && <div className="text-gray-500">Closed: {new Date(inc.closed_at).toLocaleString()}</div>}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
