import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type WidgetType = 
  | 'portfolio' 
  | 'watchlist' 
  | 'signals' 
  | 'chart' 
  | 'pnl' 
  | 'risk' 
  | 'market-overview'
  | 'trade-log'
  | 'analytics'
  | 'stats'
  | 'trading-control';

export interface WidgetSettings {
  symbol?: string;
  refreshIntervalMs?: number;
  [key: string]: any;
}

export type Widget = {
  id: string;
  type: WidgetType;
  title: string;
  x: number;
  y: number;
  w: number;
  h: number;
  minW?: number;
  minH?: number;
  visible: boolean;
  settings: WidgetSettings;
};

export interface DashboardLayout {
  version: number; // schema version for migration guard
  widgets: Widget[];
  theme: 'light' | 'dark';
  autoRefresh: boolean;
  refreshInterval: number;
  compactMode: boolean;
}

interface DashboardStore {
  layout: DashboardLayout;
  isEditing: boolean;
  selectedSymbol: string;
  
  // Actions
  setTheme: (theme: 'light' | 'dark') => void;
  toggleEditing: () => void;
  addWidget: (type: WidgetType) => void;
  removeWidget: (id: string) => void;
  updateWidget: (id: string, updates: Partial<Widget>) => void;
  updateWidgetPosition: (id: string, x: number, y: number, w: number, h: number) => void;
  setSelectedSymbol: (symbol: string) => void; // legacy global symbol (kept for backwards compat)
  updateWidgetSettings: (id: string, settings: Partial<WidgetSettings>) => void;
  resetLayout: () => void;
  exportLayout: () => string;
  importLayout: (data: string) => void;
}

const DEFAULT_SYMBOL = 'BTCUSDC';

const defaultWidgets: Widget[] = [
  {
    id: 'chart-1',
    type: 'chart',
    title: 'Price Chart',
    x: 0,
    y: 0,
    w: 8,
    h: 6,
    minW: 6,
    minH: 4,
    visible: true,
    settings: { symbol: DEFAULT_SYMBOL },
  },
  {
    id: 'watchlist-1',
    type: 'watchlist',
    title: 'Watchlist',
    x: 8,
    y: 0,
    w: 4,
    h: 6,
    minW: 3,
    minH: 3,
    visible: true,
    settings: {},
  },
  {
    id: 'signals-1',
    type: 'signals',
    title: 'Signal Feed',
    x: 0,
    y: 6,
    w: 6,
    h: 4,
    minW: 4,
    minH: 3,
    visible: true,
    settings: { symbol: DEFAULT_SYMBOL },
  },
  {
    id: 'pnl-1',
    type: 'pnl',
    title: 'P&L Overview',
    x: 6,
    y: 6,
    w: 3,
    h: 4,
    minW: 2,
    minH: 2,
    visible: true,
    settings: { symbol: DEFAULT_SYMBOL },
  },
  {
    id: 'portfolio-1',
    type: 'portfolio',
    title: 'Portfolio',
    x: 9,
    y: 6,
    w: 3,
    h: 4,
    minW: 2,
    minH: 2,
    visible: true,
    settings: { symbol: DEFAULT_SYMBOL },
  },
  {
    id: 'trading-control-1',
    type: 'trading-control',
    title: 'AI Trading Engine',
    x: 0,
    y: 10,
    w: 4,
    h: 6,
    minW: 3,
    minH: 4,
    visible: true,
    settings: {},
  },
];

const LAYOUT_VERSION = 2;

const defaultLayout: DashboardLayout = {
  version: LAYOUT_VERSION,
  widgets: defaultWidgets,
  theme: 'dark',
  autoRefresh: true,
  refreshInterval: 5000,
  compactMode: false,
};

// Migration helper for older persisted layouts
function migrateLayout(input: any): DashboardLayout {
  if (!input) return defaultLayout;
  const v = input.version ?? 1;
  if (v === LAYOUT_VERSION) return input as DashboardLayout;
  // v1 -> v2: add version + ensure each widget has settings + default symbol where missing
  if (v === 1) {
    const widgets: Widget[] = (input.widgets || []).map((w: any) => ({
      ...w,
      settings: {
        symbol: w.settings?.symbol || (['watchlist'].includes(w.type) ? undefined : DEFAULT_SYMBOL),
        refreshIntervalMs: w.settings?.refreshIntervalMs || 5000,
        ...w.settings,
      },
    }));
    return {
      version: LAYOUT_VERSION,
      widgets,
      theme: input.theme || 'dark',
      autoRefresh: input.autoRefresh ?? true,
      refreshInterval: input.refreshInterval ?? 5000,
      compactMode: input.compactMode ?? false,
    };
  }
  // Unknown future version: fall back to default with note
  console.warn('[layout] Unknown layout version; resetting to default');
  return defaultLayout;
}

export const useDashboardStore = create<DashboardStore>()(
  persist(
    (set, get) => ({
      layout: defaultLayout,
      isEditing: false,
      selectedSymbol: 'BTCUSDC',

      setTheme: (theme: 'light' | 'dark') =>
        set((state) => ({
          layout: { ...state.layout, theme },
        })),

      toggleEditing: () =>
        set((state) => ({ isEditing: !state.isEditing })),

      addWidget: (type: WidgetType) =>
        set((state) => {
          const newWidget: Widget = {
            id: `${type}-${Date.now()}`,
            type,
            title: type.charAt(0).toUpperCase() + type.slice(1),
            x: 0,
            y: 0,
            w: 4,
            h: 3,
            visible: true,
            settings: {
              symbol: ['watchlist'].includes(type) ? undefined : DEFAULT_SYMBOL,
              refreshIntervalMs: 5000,
            },
          };
          return {
            layout: {
              ...state.layout,
              widgets: [...state.layout.widgets, newWidget],
            },
          };
        }),

      removeWidget: (id: string) =>
        set((state) => ({
          layout: {
            ...state.layout,
            widgets: state.layout.widgets.filter((w) => w.id !== id),
          },
        })),

      updateWidget: (id: string, updates: Partial<Widget>) =>
        set((state) => ({
          layout: {
            ...state.layout,
            widgets: state.layout.widgets.map((w) =>
              w.id === id ? { ...w, ...updates } : w
            ),
          },
        })),

      updateWidgetSettings: (id: string, settings: Partial<WidgetSettings>) =>
        set((state) => ({
          layout: {
            ...state.layout,
            widgets: state.layout.widgets.map((w) =>
              w.id === id ? { ...w, settings: { ...w.settings, ...settings } } : w
            ),
          },
        })),

      updateWidgetPosition: (id: string, x: number, y: number, w: number, h: number) =>
        set((state) => ({
          layout: {
            ...state.layout,
            widgets: state.layout.widgets.map((widget) =>
              widget.id === id ? { ...widget, x, y, w, h } : widget
            ),
          },
        })),

      setSelectedSymbol: (symbol: string) =>
        set({ selectedSymbol: symbol }),

      resetLayout: () =>
        set({ layout: defaultLayout }),

      exportLayout: () => {
        const { layout } = get();
        const payload = { _schemaVersion: LAYOUT_VERSION, layout };
        return JSON.stringify(payload, null, 2);
      },

      importLayout: (data: string) => {
        try {
          const parsed = JSON.parse(data);
          const candidate = parsed.layout && parsed._schemaVersion ? parsed.layout : parsed;
          const migrated = migrateLayout(candidate);
          set({ layout: migrated });
        } catch (error) {
          console.error('Failed to import layout:', error);
        }
      },
    }),
    {
      name: 'quantum-trader-dashboard',
      version: 1,
      migrate: (persistedState: any) => {
        try {
          if (persistedState?.layout) {
            persistedState.layout = migrateLayout(persistedState.layout);
          }
        } catch (e) {
          console.warn('[layout] migration failed, using default', e);
          persistedState.layout = defaultLayout;
        }
        return persistedState;
      },
    }
  )
);