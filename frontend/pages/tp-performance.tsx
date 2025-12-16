// TP Performance Dashboard - Main page
import { useState, useEffect, useMemo } from 'react';
import type { TPDashboardKey, TPDashboardEntry, TPDashboardSummary, FilterState } from '@/lib/tpDashboardTypes';
import { fetchTpEntities, fetchTpEntry, fetchTpSummary } from '@/lib/tpDashboardApi';
import TpSummaryTiles from '@/components/tp/TpSummaryTiles';
import TpFilterBar from '@/components/tp/TpFilterBar';
import TpTable from '@/components/tp/TpTable';
import TpDetailDrawer from '@/components/tp/TpDetailDrawer';

export default function TpPerformancePage() {
  // Data state
  const [entities, setEntities] = useState<TPDashboardKey[]>([]);
  const [summary, setSummary] = useState<TPDashboardSummary | null>(null);
  const [selectedEntry, setSelectedEntry] = useState<TPDashboardEntry | null>(null);
  
  // UI state
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | undefined>();
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [entryLoading, setEntryLoading] = useState(false);
  const [entryError, setEntryError] = useState<string | undefined>();
  
  // Filter state
  const [filterState, setFilterState] = useState<FilterState>({
    strategy: 'all',
    symbol: 'all',
    search: '',
    showOnlyRecommendations: false,
  });

  // Fetch initial data on mount
  useEffect(() => {
    const loadInitialData = async () => {
      setLoading(true);
      setError(undefined);

      try {
        const [entitiesData, summaryData] = await Promise.all([
          fetchTpEntities(),
          fetchTpSummary(10),
        ]);

        setEntities(entitiesData);
        setSummary(summaryData);
      } catch (err) {
        const errorMessage = err instanceof Error ? err.message : 'Failed to load TP dashboard data';
        setError(errorMessage);
        console.error('Error loading TP dashboard:', err);
      } finally {
        setLoading(false);
      }
    };

    loadInitialData();
  }, []);

  // Extract unique strategies and symbols from entities
  const allStrategies = useMemo(() => {
    const strategies = Array.from(new Set(entities.map((e) => e.strategy_id))).sort();
    return strategies;
  }, [entities]);

  const allSymbols = useMemo(() => {
    const symbols = Array.from(new Set(entities.map((e) => e.symbol))).sort();
    return symbols;
  }, [entities]);

  // Filter entities based on current filter state
  const filteredEntities = useMemo(() => {
    let filtered = entities;

    // Filter by strategy
    if (filterState.strategy !== 'all') {
      filtered = filtered.filter((e) => e.strategy_id === filterState.strategy);
    }

    // Filter by symbol
    if (filterState.symbol !== 'all') {
      filtered = filtered.filter((e) => e.symbol === filterState.symbol);
    }

    // Filter by search text
    if (filterState.search.trim() !== '') {
      const searchLower = filterState.search.toLowerCase();
      filtered = filtered.filter(
        (e) =>
          e.strategy_id.toLowerCase().includes(searchLower) ||
          e.symbol.toLowerCase().includes(searchLower)
      );
    }

    // Filter by recommendations (only if checkbox is checked)
    // Note: We don't have has_recommendation in TPDashboardKey, so we skip this for now
    // If needed, we would need to fetch entries or add this field to the key

    return filtered;
  }, [entities, filterState]);

  // Handle row click - fetch entry details and open drawer
  const handleRowClick = async (key: TPDashboardKey) => {
    setDrawerOpen(true);
    setEntryLoading(true);
    setEntryError(undefined);
    setSelectedEntry(null);

    try {
      const entry = await fetchTpEntry(key.strategy_id, key.symbol);
      setSelectedEntry(entry);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to load entry details';
      setEntryError(errorMessage);
      console.error('Error loading entry:', err);
    } finally {
      setEntryLoading(false);
    }
  };

  // Handle drawer close
  const handleDrawerClose = () => {
    setDrawerOpen(false);
    setSelectedEntry(null);
    setEntryError(undefined);
  };

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-6">
        <h1 className="text-3xl font-bold text-gray-900 dark:text-gray-100">
          TP Performance Dashboard
        </h1>
        <p className="text-gray-500 dark:text-gray-400 mt-2">
          Monitor and optimize take-profit profiles across strategies and symbols
        </p>
      </div>

      {/* Loading state */}
      {loading && (
        <div className="space-y-6">
          <div className="animate-pulse">
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
              {[1, 2, 3, 4].map((i) => (
                <div key={i} className="h-24 bg-gray-200 dark:bg-slate-700 rounded-lg"></div>
              ))}
            </div>
            <div className="h-16 bg-gray-200 dark:bg-slate-700 rounded-lg mb-4"></div>
            <div className="h-64 bg-gray-200 dark:bg-slate-700 rounded-lg"></div>
          </div>
        </div>
      )}

      {/* Error state */}
      {error && !loading && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
          <h3 className="text-red-700 dark:text-red-400 font-semibold mb-2">Error Loading Dashboard</h3>
          <p className="text-red-600 dark:text-red-500">{error}</p>
        </div>
      )}

      {/* Content */}
      {!loading && !error && (
        <>
          {/* Summary Tiles */}
          <TpSummaryTiles summary={summary} loading={loading} />

          {/* Filter Bar */}
          <TpFilterBar
            strategies={allStrategies}
            symbols={allSymbols}
            filterState={filterState}
            onFilterChange={setFilterState}
          />

          {/* Table */}
          <TpTable
            entities={filteredEntities}
            loading={loading}
            onRowClick={handleRowClick}
          />

          {/* Detail Drawer */}
          <TpDetailDrawer
            entry={selectedEntry}
            open={drawerOpen}
            onClose={handleDrawerClose}
            loading={entryLoading}
            error={entryError}
          />
        </>
      )}
    </div>
  );
}
