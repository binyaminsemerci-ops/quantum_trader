import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import { Responsive, WidthProvider } from 'react-grid-layout';
import { 
  Settings, 
  Plus, 
  Eye, 
  EyeOff, 
  Download, 
  Upload, 
  RotateCcw,
  Moon,
  Sun,
  Maximize,
  Minimize
} from 'lucide-react';
import { useDashboardStore } from '../stores/dashboardStore';
import type { WidgetType } from '../stores/dashboardStore';

// Widget Components
import PriceChart from '../components/PriceChart';
import SignalFeed from '../components/SignalFeed';
import Watchlist from '../components/Watchlist';
import StressTrendsCard from '../components/StressTrendsCard';
import PortfolioWidget from '../components/PortfolioWidget';
import PnLWidget from '../components/PnLWidget';
import RiskWidget from '../components/RiskWidget';
import MarketOverviewWidget from '../components/MarketOverviewWidget';

const ResponsiveGridLayout = WidthProvider(Responsive);

const widgetComponents = {
  chart: PriceChart,
  signals: SignalFeed,
  watchlist: Watchlist,
  portfolio: PortfolioWidget,
  pnl: PnLWidget,
  risk: RiskWidget,
  'market-overview': MarketOverviewWidget,
  'trade-log': StressTrendsCard, // Placeholder
  analytics: StressTrendsCard, // Placeholder
};

const widgetIcons = {
  chart: '📊',
  signals: '🎯',
  watchlist: '👀',
  portfolio: '💼',
  pnl: '💰',
  risk: '⚠️',
  'market-overview': '🌐',
  'trade-log': '📋',
  analytics: '📈',
};

interface WidgetWrapperProps {
  widget: any;
  isEditing: boolean;
  onRemove: () => void;
  onToggleVisible: () => void;
  children: React.ReactNode;
}

const WidgetWrapper = React.memo(({ 
  widget, 
  isEditing, 
  onRemove, 
  onToggleVisible, 
  children 
}: WidgetWrapperProps) => {
  return (
    <div 
      className={`
        bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700
        ${isEditing ? 'ring-2 ring-blue-500 ring-opacity-50' : ''}
        ${!widget.visible ? 'opacity-50' : ''}
        transition-all duration-200
      `}
    >
      <div className="flex items-center justify-between p-3 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-2">
          <span className="text-lg">{widgetIcons[widget.type as WidgetType]}</span>
          <h3 className="font-semibold text-gray-900 dark:text-white">
            {widget.title}
          </h3>
        </div>
        
        {isEditing && (
          <div className="flex items-center space-x-1">
            <button
              onClick={onToggleVisible}
              className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title={widget.visible ? 'Hide widget' : 'Show widget'}
            >
              {widget.visible ? (
                <Eye className="w-4 h-4 text-gray-600 dark:text-gray-300" />
              ) : (
                <EyeOff className="w-4 h-4 text-gray-400" />
              )}
            </button>
            <button
              onClick={onRemove}
              className="p-1 rounded hover:bg-red-100 dark:hover:bg-red-900 transition-colors"
              title="Remove widget"
            >
              <span className="text-red-500">×</span>
            </button>
          </div>
        )}
      </div>
      
      <div className="p-4 h-full">
        {children}
      </div>
    </div>
  );
});

export default function SuperDashboard() {
  // Use selectors to reduce re-renders
  const layout = useDashboardStore(state => state.layout);
  const isEditing = useDashboardStore(state => state.isEditing);
  const selectedSymbol = useDashboardStore(state => state.selectedSymbol);
  const setTheme = useDashboardStore(state => state.setTheme);
  const toggleEditing = useDashboardStore(state => state.toggleEditing);
  const addWidget = useDashboardStore(state => state.addWidget);
  const removeWidget = useDashboardStore(state => state.removeWidget);
  const updateWidget = useDashboardStore(state => state.updateWidget);
  const updateWidgetPosition = useDashboardStore(state => state.updateWidgetPosition);
  const setSelectedSymbol = useDashboardStore(state => state.setSelectedSymbol);
  const resetLayout = useDashboardStore(state => state.resetLayout);
  const exportLayout = useDashboardStore(state => state.exportLayout);
  const importLayout = useDashboardStore(state => state.importLayout);

  const [showAddMenu, setShowAddMenu] = useState(false);
  const [compactMode, setCompactMode] = useState(false);

  const layouts = useMemo(() => ({
    lg: layout.widgets.map(w => ({
      i: w.id,
      x: w.x,
      y: w.y,
      w: w.w,
      h: w.h,
      minW: w.minW,
      minH: w.minH,
      static: !isEditing
    }))
  }), [layout.widgets, isEditing]);

  // Debounced layout change to prevent excessive re-renders
  const debounceTimeoutRef = useRef<NodeJS.Timeout>();
  
  const handleLayoutChange = useCallback((newLayout: any) => {
    // Clear previous timeout
    if (debounceTimeoutRef.current) {
      clearTimeout(debounceTimeoutRef.current);
    }
    
    // Debounce the update to prevent flashing
    debounceTimeoutRef.current = setTimeout(() => {
      newLayout.forEach((item: any) => {
        updateWidgetPosition(item.i, item.x, item.y, item.w, item.h);
      });
    }, 100); // 100ms debounce
  }, [updateWidgetPosition]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (debounceTimeoutRef.current) {
        clearTimeout(debounceTimeoutRef.current);
      }
    };
  }, []);

  const handleAddWidget = (type: WidgetType) => {
    addWidget(type);
    setShowAddMenu(false);
  };

  const handleExport = () => {
    const data = exportLayout();
    const blob = new Blob([data], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'dashboard-layout.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleImport = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.json';
    input.onchange = (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const data = e.target?.result as string;
          importLayout(data);
        };
        reader.readAsText(file);
      }
    };
    input.click();
  };

  const availableWidgets: WidgetType[] = [
    'chart', 'signals', 'watchlist', 'portfolio', 
    'pnl', 'risk', 'market-overview', 'trade-log', 'analytics'
  ];

  return (
    <div className={`min-h-screen ${layout.theme === 'dark' ? 'dark bg-gray-900' : 'bg-gray-50'}`}>
      {/* Top Toolbar */}
      <div className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
        <div className="px-4 py-3 flex items-center justify-between">
          <div className="flex items-center space-x-4">
            <h1 className="text-xl font-bold text-gray-900 dark:text-white">
              🚀 Quantum Trader Pro
            </h1>
            
            {/* Symbol Selector */}
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              title="Select trading symbol"
              className="px-3 py-1 rounded border border-gray-300 dark:border-gray-600 
                       bg-white dark:bg-gray-700 text-gray-900 dark:text-white
                       focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            >
              <option value="BTCUSDT">BTC/USDT</option>
              <option value="ETHUSDT">ETH/USDT</option>
              <option value="SOLUSDT">SOL/USDT</option>
              <option value="ADAUSDT">ADA/USDT</option>
            </select>
          </div>

          {/* Toolbar Actions */}
          <div className="flex items-center space-x-2">
            {/* Theme Toggle */}
            <button
              onClick={() => setTheme(layout.theme === 'dark' ? 'light' : 'dark')}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="Toggle theme"
            >
              {layout.theme === 'dark' ? (
                <Sun className="w-5 h-5 text-yellow-500" />
              ) : (
                <Moon className="w-5 h-5 text-gray-600" />
              )}
            </button>

            {/* Compact Mode */}
            <button
              onClick={() => setCompactMode(!compactMode)}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              title="Toggle compact mode"
            >
              {compactMode ? (
                <Maximize className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              ) : (
                <Minimize className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              )}
            </button>

            {/* Add Widget */}
            <div className="relative">
              <button
                onClick={() => setShowAddMenu(!showAddMenu)}
                className="p-2 rounded-lg bg-blue-500 hover:bg-blue-600 text-white transition-colors"
                title="Add widget"
              >
                <Plus className="w-5 h-5" />
              </button>

              {showAddMenu && (
                <div className="absolute right-0 mt-2 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 z-50">
                  <div className="p-2">
                    {availableWidgets.map((type) => (
                      <button
                        key={type}
                        onClick={() => handleAddWidget(type)}
                        className="w-full text-left px-3 py-2 rounded hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors flex items-center space-x-2"
                      >
                        <span>{widgetIcons[type]}</span>
                        <span className="capitalize text-gray-700 dark:text-gray-300">
                          {type.replace('-', ' ')}
                        </span>
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>

            {/* Settings Menu */}
            <div className="flex items-center space-x-1">
              <button
                onClick={handleExport}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                title="Export layout"
              >
                <Download className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              </button>

              <button
                onClick={handleImport}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                title="Import layout"
              >
                <Upload className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              </button>

              <button
                onClick={resetLayout}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                title="Reset layout"
              >
                <RotateCcw className="w-5 h-5 text-gray-600 dark:text-gray-400" />
              </button>

              <button
                onClick={toggleEditing}
                className={`p-2 rounded-lg transition-colors ${
                  isEditing 
                    ? 'bg-blue-500 text-white' 
                    : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-600 dark:text-gray-400'
                }`}
                title="Toggle edit mode"
              >
                <Settings className="w-5 h-5" />
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Dashboard Grid */}
      <div className="p-4">
        <ResponsiveGridLayout
          className="layout"
          layouts={layouts}
          onLayoutChange={handleLayoutChange}
          breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
          cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
          rowHeight={compactMode ? 40 : 60}
          isDraggable={isEditing}
          isResizable={isEditing}
          compactType={compactMode ? 'vertical' : null}
          margin={[16, 16]}
        >
          {useMemo(() => 
            layout.widgets
              .filter(widget => widget.visible)
              .map((widget) => {
                const WidgetComponent = widgetComponents[widget.type as keyof typeof widgetComponents];
                
                return (
                  <div key={widget.id}>
                    <WidgetWrapper
                      widget={widget}
                      isEditing={isEditing}
                      onRemove={useCallback(() => removeWidget(widget.id), [widget.id, removeWidget])}
                      onToggleVisible={useCallback(() => updateWidget(widget.id, { visible: !widget.visible }), [widget.id, widget.visible, updateWidget])}
                    >
                      {WidgetComponent ? (
                        <WidgetComponent symbol={selectedSymbol} {...(widget.settings || {})} />
                      ) : (
                        <div className="flex items-center justify-center h-full text-gray-500 dark:text-gray-400">
                          <span>Widget: {widget.type}</span>
                        </div>
                      )}
                    </WidgetWrapper>
                  </div>
                );
              }), [layout.widgets, isEditing, selectedSymbol, removeWidget, updateWidget])}
        </ResponsiveGridLayout>
      </div>

      {/* Edit Mode Overlay */}
      {isEditing && (
        <div className="fixed bottom-4 right-4 bg-blue-500 text-white px-4 py-2 rounded-lg shadow-lg">
          📝 Edit Mode Active - Drag & resize widgets
        </div>
      )}
    </div>
  );
}