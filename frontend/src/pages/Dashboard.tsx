import React, { useCallback, useMemo } from 'react';
import { Responsive, WidthProvider } from 'react-grid-layout';
import { useDashboardStore } from '../stores/dashboardStore';
import { lazy, Suspense, useState } from 'react';
const PortfolioWidget = lazy(() => import('../components/PortfolioWidget'));
const PnLWidget = lazy(() => import('../components/PnLWidget'));
const RiskWidget = lazy(() => import('../components/RiskWidget'));
const RiskDashboard = lazy(() => import('../components/RiskDashboard'));
const MarketOverviewWidget = lazy(() => import('../components/MarketOverviewWidget'));
const ChartWidget = lazy(() => import('../components/ChartWidget'));
const WatchlistWidget = lazy(() => import('../components/WatchlistWidget'));
const SignalsWidget = lazy(() => import('../components/SignalsWidget'));
const StatsWidget = lazy(() => import('../components/StatsWidget'));
const TradingControlWidget = lazy(() => import('../components/TradingControlWidget'));
const StatusWidget = lazy(() => import('../components/StatusWidget'));
import { useDebouncedLayoutSync } from '../hooks/useDebouncedLayoutSync';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import '../styles/dashboard.css'; // optional if exists

const ResponsiveGridLayout = WidthProvider(Responsive);

// Map widget type to component
const widgetRenderer: Record<string, React.FC<any>> = {
  portfolio: (p) => <PortfolioWidget {...p} />,
  pnl: (p) => <PnLWidget {...p} />,
  risk: (p) => <RiskWidget {...p} />,
  'risk-dashboard': () => <RiskDashboard />,
  'market-overview': (p) => <MarketOverviewWidget {...p} />,
  chart: (p) => <ChartWidget {...p} />,
  watchlist: () => <WatchlistWidget />,
  signals: () => <SignalsWidget />,
  stats: () => <StatsWidget />,
  'trading-control': () => <TradingControlWidget />,
  'system-status': () => <StatusWidget />,
};

const placeholder = (type: string) => (
  <div className="h-full flex items-center justify-center text-xs text-gray-500 dark:text-gray-400">
    <span>Widget '{type}' ikke implementert ennå</span>
  </div>
);

export default function Dashboard() {
  const layoutState = useDashboardStore(s => s.layout);
  const isEditing = useDashboardStore(s => s.isEditing);
  const toggleEditing = useDashboardStore(s => s.toggleEditing);
  const setTheme = useDashboardStore(s => s.setTheme);
  const addWidget = useDashboardStore(s => s.addWidget);
  const selectedSymbol = useDashboardStore(s => s.selectedSymbol);
  const setSelectedSymbol = useDashboardStore(s => s.setSelectedSymbol);
  const removeWidget = useDashboardStore(s => s.removeWidget);
  const updateWidgetPosition = useDashboardStore(s => s.updateWidgetPosition);
  const updateWidgetSettings = useDashboardStore(s => s.updateWidgetSettings);
  const [persisting, setPersisting] = useState(false);
  const [persistStatus, setPersistStatus] = useState<string>('');
  const resetLayout = useDashboardStore(s => s.resetLayout);
  const exportLayout = useDashboardStore(s => s.exportLayout);
  const importLayout = useDashboardStore(s => s.importLayout);
  // Auto sync layout to backend (debounced). In future we can guard with feature flag or auth token.
  useDebouncedLayoutSync(2000, true);

  const currentTheme = layoutState.theme;

  const gridItems = useMemo(() => layoutState.widgets.filter(w => w.visible), [layoutState.widgets]);

  const onLayoutChange = useCallback((newLayout: any[]) => {
    if (!isEditing) return; // only persist when editing
    newLayout.forEach(l => {
      const w = gridItems.find(g => g.id === l.i);
      if (w && (w.x !== l.x || w.y !== l.y || w.w !== l.w || w.h !== l.h)) {
        updateWidgetPosition(w.id, l.x, l.y, l.w, l.h);
      }
    });
  }, [isEditing, gridItems, updateWidgetPosition]);

  const handleAdd = (type: string) => addWidget(type as any);

  const handleExport = () => {
    const data = exportLayout();
    navigator.clipboard?.writeText(data).catch(() => {});
    // eslint-disable-next-line no-alert
    alert('Layout kopiert til utklippstavle');
  };

  const handleImport = () => {
    const data = window.prompt('Lim inn eksportert layout JSON:');
    if (data) importLayout(data);
  };

  async function persistToBackend() {
    setPersisting(true);
    setPersistStatus('');
    try {
      const payload = JSON.parse(exportLayout());
      const res = await fetch('/api/layout', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ layout: payload.layout, version: payload.layout.version, _schemaVersion: payload._schemaVersion }) });
      if (!res.ok) throw new Error(String(res.status));
      setPersistStatus('Lagret');
    } catch (e:any) {
      setPersistStatus('Feil: ' + e.message);
    } finally {
      setPersisting(false);
      setTimeout(()=> setPersistStatus(''), 4000);
    }
  }

  async function loadFromBackend() {
    try {
      const res = await fetch('/api/layout');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();
      if (data?.layout) {
        importLayout(JSON.stringify({ _schemaVersion: data._schemaVersion || data.version || 2, layout: data.layout }));
      } else { alert('Ingen layout på server.'); }
    } catch (e:any) {
      alert('Feil ved henting: ' + e.message);
    }
  }

  return (
    <div className={currentTheme === 'dark' ? 'dark bg-gray-900 text-gray-100 min-h-screen' : 'bg-gray-50 text-gray-900 min-h-screen'}>
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 flex flex-wrap gap-2 items-center">
        <h1 className="text-lg font-semibold">Quantum Trader Dashboard</h1>
        <div className="flex gap-2 items-center text-sm">
          <select
            aria-label="Velg symbol"
            value={selectedSymbol}
            onChange={e => setSelectedSymbol(e.target.value)}
            className="px-2 py-1 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:outline-none focus:ring focus:ring-indigo-500/30 text-xs"
          >
            {['BTCUSDC','ETHUSDC','SOLUSDC','ADAUSDC','XRPUSDC'].map(sym => (
              <option key={sym} value={sym}>{sym}</option>
            ))}
          </select>
          <button onClick={() => setTheme(currentTheme === 'dark' ? 'light' : 'dark')} className="px-3 py-1 rounded bg-indigo-600 text-white hover:bg-indigo-700">Tema: {currentTheme}</button>
          <button onClick={toggleEditing} className="px-3 py-1 rounded bg-blue-600 text-white hover:bg-blue-700">{isEditing ? 'Lås' : 'Rediger'}</button>
          {isEditing && (
            <>
              <button onClick={() => handleAdd('pnl')} className="px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700">+ PnL</button>
              <button onClick={() => handleAdd('portfolio')} className="px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700">+ Portefølje</button>
              <button onClick={() => handleAdd('risk')} className="px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700">+ Risiko</button>
              <button onClick={() => handleAdd('risk-dashboard')} className="px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700">+ Risiko Pro</button>
              <button onClick={() => handleAdd('market-overview')} className="px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700">+ Marked</button>
              <button onClick={() => handleAdd('chart')} className="px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700">+ Chart</button>
              <button onClick={() => handleAdd('watchlist')} className="px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700">+ Watchlist</button>
              <button onClick={() => handleAdd('signals')} className="px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700">+ Signals</button>
              <button onClick={() => handleAdd('stats')} className="px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700">+ Stats</button>
              <button onClick={() => handleAdd('system-status')} className="px-3 py-1 rounded bg-green-600 text-white hover:bg-green-700">+ Status</button>
              <button onClick={resetLayout} className="px-3 py-1 rounded bg-yellow-600 text-white hover:bg-yellow-700">Reset</button>
              <button onClick={handleExport} className="px-3 py-1 rounded bg-gray-600 text-white hover:bg-gray-700">Export</button>
              <button onClick={handleImport} className="px-3 py-1 rounded bg-gray-600 text-white hover:bg-gray-700">Import</button>
              <button onClick={persistToBackend} disabled={persisting} className="px-3 py-1 rounded bg-purple-600 text-white hover:bg-purple-700 disabled:opacity-50">{persisting ? 'Lagrer...' : 'Sync->Server'}</button>
              <button onClick={loadFromBackend} className="px-3 py-1 rounded bg-purple-600 text-white hover:bg-purple-700">Hent Server</button>
              {persistStatus && <span className="text-xs text-gray-500 dark:text-gray-400">{persistStatus}</span>}
            </>
          )}
        </div>
      </div>
      <div className="p-4">
        <ResponsiveGridLayout
          className="layout"
          rowHeight={30}
          breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
          cols={{ lg: 12, md: 10, sm: 8, xs: 6, xxs: 4 }}
          isDraggable={isEditing}
            isResizable={isEditing}
          margin={[12, 12]}
          onLayoutChange={onLayoutChange}
          compactType={layoutState.compactMode ? 'vertical' : null}
          draggableCancel=".nodrag"
        >
          {gridItems.map(w => (
            <div key={w.id} data-grid={{ i: w.id, x: w.x, y: w.y, w: w.w, h: w.h, minW: w.minW, minH: w.minH }} className="rounded-lg border border-gray-300 dark:border-gray-700 bg-white dark:bg-gray-800 overflow-hidden flex flex-col">
              <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 text-sm font-medium select-none">
                <span className="truncate">{w.title}</span>
                {isEditing && (
                  <div className="flex items-center gap-2">
                    {w.settings && 'symbol' in w.settings && (
                      <select
                        aria-label="Widget symbol"
                        value={w.settings.symbol || ''}
                        onChange={e => updateWidgetSettings(w.id, { symbol: e.target.value })}
                        className="px-1 py-0.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 text-[11px] focus:outline-none"
                      >
                        {['BTCUSDC','ETHUSDC','SOLUSDC','ADAUSDC','XRPUSDC'].map(sym => (
                          <option key={sym} value={sym}>{sym}</option>
                        ))}
                      </select>
                    )}
                    <button onClick={() => removeWidget(w.id)} className="text-red-500 hover:text-red-600" aria-label="Remove">✕</button>
                  </div>
                )}
              </div>
              <div className="p-3 flex-1 min-h-0">
                <Suspense fallback={<div className="text-xs text-gray-500">Laster...</div>}>
                  {(widgetRenderer[w.type] || (() => placeholder(w.type)))({ symbol: w.settings?.symbol || selectedSymbol })}
                  {isEditing && w.settings?.refreshIntervalMs !== undefined && (
                    <div className="mt-2 flex items-center gap-1 text-[11px] text-gray-500">
                      <label className="flex items-center gap-1">Refresh(ms)
                        <input
                          type="number"
                          className="w-20 px-1 py-0.5 rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-800 focus:outline-none"
                          value={w.settings.refreshIntervalMs}
                          min={1000}
                          step={500}
                          onChange={e => updateWidgetSettings(w.id, { refreshIntervalMs: Number(e.target.value) })}
                        />
                      </label>
                    </div>
                  )}
                </Suspense>
              </div>
            </div>
          ))}
        </ResponsiveGridLayout>
      </div>
    </div>
  );
}
