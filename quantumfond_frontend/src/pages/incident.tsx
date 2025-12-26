import { useEffect, useState } from 'react';

export default function Incident() {
  const [incidents, setIncidents] = useState<any>(null);

  useEffect(() => {
    fetch('http://localhost:8000/incidents/')
      .then(res => res.json())
      .then(data => setIncidents(data))
      .catch(err => console.error(err));
  }, []);

  const getSeverityColor = (severity: string) => {
    switch (severity) {
      case 'critical':
        return 'bg-red-900/30 text-red-400';
      case 'high':
        return 'bg-orange-900/30 text-orange-400';
      case 'medium':
        return 'bg-yellow-900/30 text-yellow-400';
      case 'low':
      default:
        return 'bg-blue-900/30 text-blue-400';
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold text-white">Incident Management</h1>
        <button className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg text-white font-semibold transition-colors">
          + New Incident
        </button>
      </div>

      <div className="bg-gray-900 border border-gray-800 rounded-lg overflow-hidden">
        <table className="w-full">
          <thead className="bg-gray-950">
            <tr>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">ID</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Title</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Severity</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Status</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Assigned To</th>
              <th className="px-6 py-3 text-left text-xs font-medium text-gray-400 uppercase">Created</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-800">
            {incidents?.incidents?.map((incident: any) => (
              <tr key={incident.id} className="hover:bg-gray-800/50 cursor-pointer">
                <td className="px-6 py-4 text-sm text-gray-400">#{incident.id}</td>
                <td className="px-6 py-4 text-sm font-medium text-white">{incident.title}</td>
                <td className="px-6 py-4 text-sm">
                  <span className={`px-2 py-1 rounded text-xs font-semibold ${getSeverityColor(incident.severity)}`}>
                    {incident.severity}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm">
                  <span className="px-2 py-1 rounded text-xs font-semibold bg-gray-700 text-gray-300">
                    {incident.status}
                  </span>
                </td>
                <td className="px-6 py-4 text-sm text-gray-300">{incident.assigned_to}</td>
                <td className="px-6 py-4 text-sm text-gray-400">
                  {incident.created_at ? new Date(incident.created_at).toLocaleDateString() : 'N/A'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
