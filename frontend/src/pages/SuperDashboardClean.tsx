import { useState } from 'react';
import { useDashboardStore } from '../stores/dashboardStore';

export default function SuperDashboard() {
  // Use selectors to reduce re-renders
  const layout = useDashboardStore(state => state.layout);
  const isEditing = useDashboardStore(state => state.isEditing);
  const selectedSymbol = useDashboardStore(state => state.selectedSymbol);
  const setTheme = useDashboardStore(state => state.setTheme);
  const addWidget = useDashboardStore(state => state.addWidget);

  const [debugAdded, setDebugAdded] = useState(0);
  const handleAddWidget = () => { addWidget('pnl'); setDebugAdded(n => n+1); };

  return (
    <div className={`min-h-screen p-6 space-y-6 ${layout.theme === 'dark' ? 'dark bg-gray-900 text-gray-100' : 'bg-gray-50 text-gray-900'}`}>
      <h1 className="text-3xl font-bold">Debug Dashboard Render</h1>
      <ul className="text-sm font-mono bg-white dark:bg-gray-800 rounded-lg p-4 shadow border border-gray-200 dark:border-gray-700">
        <li>widgets: {layout.widgets.length}</li>
        <li>theme: {layout.theme}</li>
        <li>editing: {String(isEditing)}</li>
        <li>selectedSymbol: {selectedSymbol}</li>
        <li>store ok: {layout && typeof layout === 'object' ? 'yes' : 'no'}</li>
      </ul>
      <button
        onClick={() => setTheme(layout.theme === 'dark' ? 'light' : 'dark')}
        className="px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700"
      >Toggle Theme</button>
      <button
        onClick={handleAddWidget}
        className="px-4 py-2 rounded bg-green-600 text-white hover:bg-green-700"
      >Add PnL Widget (added {debugAdded})</button>
      <div className="text-xs opacity-70">(Grid midlertidig deaktivert for feilsøking)</div>
    </div>
  );
}