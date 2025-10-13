import { useState } from 'react';

interface DashboardSettingsProps {
  onLayoutChange: (layout: string) => void;
  onRefreshIntervalChange: (interval: number) => void;
  currentLayout: string;
  currentInterval: number;
}

export default function DashboardSettings({ 
  onLayoutChange, 
  onRefreshIntervalChange, 
  currentLayout, 
  currentInterval 
}: DashboardSettingsProps): JSX.Element {
  const [isOpen, setIsOpen] = useState(false);

  const layouts = [
    { id: 'compact', name: 'Compact', icon: '📱' },
    { id: 'balanced', name: 'Balanced', icon: '⚖️' },
    { id: 'expanded', name: 'Expanded', icon: '📺' },
    { id: 'trading', name: 'Trading Focus', icon: '💹' }
  ];

  const refreshIntervals = [
    { value: 1000, label: '1s (High Frequency)' },
    { value: 5000, label: '5s (Balanced)' },
    { value: 10000, label: '10s (Conservative)' },
    { value: 30000, label: '30s (Low Bandwidth)' }
  ];

  return (
    <div className="relative">
      {/* Settings Toggle Button */}
      <button 
        onClick={() => setIsOpen(!isOpen)}
        className="px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg font-medium transition-all shadow-md hover:shadow-lg flex items-center gap-2"
      >
        <span>⚙️</span>
        <span className="hidden sm:inline">Settings</span>
      </button>

      {/* Settings Dropdown */}
      {isOpen && (
        <div className="absolute right-0 top-12 w-80 bg-white dark:bg-gray-800 border dark:border-gray-700 rounded-xl shadow-xl z-50 p-4">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-800 dark:text-white">Dashboard Settings</h3>
            <button 
              onClick={() => setIsOpen(false)}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
            >
              ✕
            </button>
          </div>

          {/* Layout Selection */}
          <div className="mb-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Layout Style
            </label>
            <div className="grid grid-cols-2 gap-2">
              {layouts.map(layout => (
                <button
                  key={layout.id}
                  onClick={() => onLayoutChange(layout.id)}
                  className={`p-3 rounded-lg border-2 transition-all text-sm font-medium ${
                    currentLayout === layout.id 
                      ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/20 text-blue-700 dark:text-blue-300' 
                      : 'border-gray-200 dark:border-gray-600 hover:border-gray-300 dark:hover:border-gray-500'
                  }`}
                >
                  <div className="text-lg mb-1">{layout.icon}</div>
                  <div>{layout.name}</div>
                </button>
              ))}
            </div>
          </div>

          {/* Refresh Interval */}
          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Data Refresh Rate
            </label>
            <select 
              value={currentInterval} 
              onChange={(e) => onRefreshIntervalChange(Number(e.target.value))}
              title="Select data refresh rate"
              className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-white"
            >
              {refreshIntervals.map(interval => (
                <option key={interval.value} value={interval.value}>
                  {interval.label}
                </option>
              ))}
            </select>
          </div>

          {/* Quick Actions */}
          <div className="border-t dark:border-gray-700 pt-4">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Quick Actions
            </label>
            <div className="flex gap-2">
              <button className="flex-1 px-3 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg text-sm font-medium transition-all">
                📊 Export Data
              </button>
              <button className="flex-1 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm font-medium transition-all">
                🔄 Reset Layout
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}