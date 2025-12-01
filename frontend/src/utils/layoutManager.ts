// Layout configurations for different dashboard modes
export const layoutConfigs = {
  compact: {
    gridCols: 'grid-cols-1 sm:grid-cols-2 mdp:grid-cols-3',
    panelVariant: 'compact' as const,
    defaultExpanded: false,
    gap: 'gap-3',
    padding: 'p-2'
  },
  balanced: {
    gridCols: 'grid-cols-1 sm:grid-cols-2 mdp:grid-cols-3 xl:grid-cols-4',
    panelVariant: 'default' as const,
    defaultExpanded: true,
    gap: 'gap-6',
    padding: 'p-4'
  },
  expanded: {
    gridCols: 'grid-cols-1 lg:grid-cols-2 xl:grid-cols-3',
    panelVariant: 'default' as const,
    defaultExpanded: true,
    gap: 'gap-8',
    padding: 'p-6'
  },
  trading: {
    gridCols: 'grid-cols-1 lg:grid-cols-2',
    panelVariant: 'default' as const,
    defaultExpanded: true,
    gap: 'gap-4',
    padding: 'p-4'
  }
};

export type LayoutType = keyof typeof layoutConfigs;

export const getLayoutConfig = (layout: LayoutType) => {
  return layoutConfigs[layout] || layoutConfigs.balanced;
};

// Priority order for different layouts
export const componentPriorities = {
  compact: ['AITradingMonitor', 'PnLTracker', 'StatsCard', 'CoinPriceMonitor'],
  balanced: ['AITradingMonitor', 'PnLTracker', 'CoinPriceMonitor', 'AdvancedTechnicalAnalysis', 'StatsCard', 'RiskCards'],
  expanded: ['AITradingMonitor', 'AdvancedTechnicalAnalysis', 'PnLTracker', 'CoinPriceMonitor', 'StatsCard', 'RiskCards', 'AnalyticsCards'],
  trading: ['AITradingMonitor', 'PnLTracker', 'PriceChart', 'TradeTable', 'AdvancedTechnicalAnalysis']
};