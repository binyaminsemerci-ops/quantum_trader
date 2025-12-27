// TP Filter Bar - filters for strategy, symbol, search, and recommendations
import type { FilterState } from '@/lib/tpDashboardTypes';

interface TpFilterBarProps {
  strategies: string[];
  symbols: string[];
  filterState: FilterState;
  onFilterChange: (nextState: FilterState) => void;
}

export default function TpFilterBar({
  strategies = [],
  symbols = [],
  filterState,
  onFilterChange,
}: TpFilterBarProps) {
  const handleStrategyChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onFilterChange({
      ...filterState,
      strategy: e.target.value || 'all',
    });
  };

  const handleSymbolChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onFilterChange({
      ...filterState,
      symbol: e.target.value || 'all',
    });
  };

  const handleSearchChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFilterChange({
      ...filterState,
      search: e.target.value,
    });
  };

  const handleRecommendationToggle = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFilterChange({
      ...filterState,
      onlyWithRecommendation: e.target.checked,
    });
  };

  return (
    <div className="bg-white dark:bg-slate-800 rounded-lg p-4 mb-4 border border-gray-200 dark:border-slate-700">
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        {/* Strategy dropdown */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Strategy
          </label>
          <select
            value={filterState.strategyId || ''}
            onChange={handleStrategyChange}
            className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-gray-300 dark:border-slate-600 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="all">All Strategies</option>
            {strategies.map((strategy) => (
              <option key={strategy} value={strategy}>
                {strategy}
              </option>
            ))}
          </select>
        </div>

        {/* Symbol dropdown */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Symbol
          </label>
          <select
            value={filterState.symbol}
            onChange={handleSymbolChange}
            className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-gray-300 dark:border-slate-600 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
          >
            <option value="all">All Symbols</option>
            {symbols.map((symbol) => (
              <option key={symbol} value={symbol}>
                {symbol}
              </option>
            ))}
          </select>
        </div>

        {/* Search input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Search
          </label>
          <input
            type="text"
            value={filterState.search}
            onChange={handleSearchChange}
            placeholder="Search strategy or symbol..."
            className="w-full px-3 py-2 bg-white dark:bg-slate-900 border border-gray-300 dark:border-slate-600 rounded-md focus:outline-none focus:ring-2 focus:ring-primary"
          />
        </div>

        {/* Recommendation toggle */}
        <div className="flex items-end">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={filterState.onlyWithRecommendation}
              onChange={handleRecommendationToggle}
              className="w-4 h-4 text-primary bg-white dark:bg-slate-900 border-gray-300 dark:border-slate-600 rounded focus:ring-primary"
            />
            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Only with recommendations
            </span>
          </label>
        </div>
      </div>
    </div>
  );
}
