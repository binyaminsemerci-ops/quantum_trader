/**
 * Auto Layout Manager - Intelligent Dashboard Layout Auto-Repair
 * Automatisk fikser layout problemer uten manuell inngripen!
 */

import { dashboardHealthMonitor, type DashboardHealthReport } from './dashboardHealthMonitor';

export interface LayoutPreset {
  id: string;
  name: string;
  description: string;
  config: LayoutConfig;
}

export interface LayoutConfig {
  gridCols: {
    base: string;
    sm: string;
    md: string;
    lg: string;
    xl: string;
  };
  components: ComponentLayout[];
}

export interface ComponentLayout {
  id: string;
  position: number;
  colSpan: {
    base: number;
    sm: number;
    md: number;
    lg: number;
    xl: number;
  };
  component: string;
  wrapInPanel?: boolean;
  panelConfig?: {
    title: string;
    icon: string;
    variant: 'default' | 'compact' | 'minimal';
    defaultExpanded: boolean;
  };
}

class AutoLayoutManager {
  private static instance: AutoLayoutManager;
  private layoutPresets: Map<string, LayoutPreset> = new Map();
  private currentLayout?: string;
  private autoFixEnabled: boolean = true;

  private constructor() {
    this.setupDefaultLayouts();
    this.setupEventListeners();
  }

  public static getInstance(): AutoLayoutManager {
    if (!AutoLayoutManager.instance) {
      AutoLayoutManager.instance = new AutoLayoutManager();
    }
    return AutoLayoutManager.instance;
  }

  private setupDefaultLayouts() {
    // Optimal Layout - Basert på dine krav
    this.layoutPresets.set('optimal', {
      id: 'optimal',
      name: 'Optimal Layout',
      description: 'The best layout with Market Candles properly positioned and Trade History at full width',
      config: {
        gridCols: {
          base: 'grid-cols-1',
          sm: 'sm:grid-cols-2',
          md: 'md:grid-cols-3',
          lg: 'lg:grid-cols-4',
          xl: 'xl:grid-cols-4'
        },
        components: [
          // Row 1 - Core Components
          {
            id: 'ai-monitor',
            position: 0,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'AITradingMonitor'
          },
          {
            id: 'pnl-tracker',
            position: 1,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'PnLTracker'
          },
          {
            id: 'price-monitor',
            position: 2,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'CoinPriceMonitor'
          },
          
          // Row 2 - Technical Analysis
          {
            id: 'technical-analysis',
            position: 3,
            colSpan: { base: 1, sm: 2, md: 2, lg: 2, xl: 2 },
            component: 'AdvancedTechnicalAnalysis'
          },
          {
            id: 'equity-chart',
            position: 4,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'EquityChart',
            wrapInPanel: true,
            panelConfig: {
              title: 'Equity Curve',
              icon: '📈',
              variant: 'default',
              defaultExpanded: true
            }
          },
          {
            id: 'watchlist',
            position: 5,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'Watchlist',
            wrapInPanel: true,
            panelConfig: {
              title: 'Watchlist',
              icon: '👁',
              variant: 'default',
              defaultExpanded: true
            }
          },

          // Row 3 - Metrics Cards
          {
            id: 'stats-card',
            position: 6,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'StatsCard'
          },
          {
            id: 'risk-cards',
            position: 7,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'RiskCards'
          },
          {
            id: 'analytics-cards',
            position: 8,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'AnalyticsCards'
          },
          {
            id: 'trade-logs',
            position: 9,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'TradeLogs',
            wrapInPanel: true,
            panelConfig: {
              title: 'Activity Logs',
              icon: '📋',
              variant: 'compact',
              defaultExpanded: false
            }
          },

          // Row 4 - Charts (FIXED: Market Candles NOT in header!)
          {
            id: 'live-price-action',
            position: 10,
            colSpan: { base: 1, sm: 2, md: 2, lg: 2, xl: 2 },
            component: 'PriceChart',
            wrapInPanel: true,
            panelConfig: {
              title: 'Live Price Action',
              icon: '📈',
              variant: 'default',
              defaultExpanded: true
            }
          },
          {
            id: 'market-candles',
            position: 11,
            colSpan: { base: 1, sm: 2, md: 2, lg: 2, xl: 2 },
            component: 'CandlesChart',
            wrapInPanel: true,
            panelConfig: {
              title: 'Market Candles',
              icon: '🕯️',
              variant: 'default',
              defaultExpanded: true
            }
          },

          // Row 5 - Full Width Trading Activity (FIXED: Full width!)
          {
            id: 'trade-history',
            position: 12,
            colSpan: { base: 1, sm: 2, md: 3, lg: 4, xl: 4 },
            component: 'TradeTable',
            wrapInPanel: true,
            panelConfig: {
              title: 'Trade History',
              icon: '🔄',
              variant: 'default',
              defaultExpanded: true
            }
          }
        ]
      }
    });

    // Compact Layout
    this.layoutPresets.set('compact', {
      id: 'compact',
      name: 'Compact Layout',
      description: 'Space-efficient layout for smaller screens',
      config: {
        gridCols: {
          base: 'grid-cols-1',
          sm: 'sm:grid-cols-2',
          md: 'md:grid-cols-2',
          lg: 'lg:grid-cols-3',
          xl: 'xl:grid-cols-3'
        },
        components: [
          // More compact arrangement
          {
            id: 'ai-monitor',
            position: 0,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'AITradingMonitor'
          },
          {
            id: 'pnl-tracker',
            position: 1,
            colSpan: { base: 1, sm: 1, md: 1, lg: 1, xl: 1 },
            component: 'PnLTracker'
          },
          {
            id: 'price-monitor',
            position: 2,
            colSpan: { base: 1, sm: 2, md: 2, lg: 1, xl: 1 },
            component: 'CoinPriceMonitor'
          },
          {
            id: 'trade-history',
            position: 3,
            colSpan: { base: 1, sm: 2, md: 2, lg: 3, xl: 3 },
            component: 'TradeTable',
            wrapInPanel: true,
            panelConfig: {
              title: 'Trade History',
              icon: '🔄',
              variant: 'compact',
              defaultExpanded: true
            }
          }
        ]
      }
    });

    this.currentLayout = 'optimal'; // Default to optimal layout
  }

  private setupEventListeners() {
    // Listen for repair events from health monitor
    window.addEventListener('dashboard:repair:container', () => this.repairDashboardContainer());
    window.addEventListener('dashboard:repair:grid', () => this.repairGridStructure());
    window.addEventListener('dashboard:repair:candles', () => this.repairCandlesPosition());
    window.addEventListener('dashboard:repair:tradehistory', () => this.repairTradeHistoryWidth());
    window.addEventListener('dashboard:repair:pricesync', () => this.repairPriceSync());
    window.addEventListener('dashboard:repair:apiconnection', () => this.repairApiConnection());

    // Monitor layout changes
    const observer = new MutationObserver(() => {
      if (this.autoFixEnabled) {
        this.debounceLayoutCheck();
      }
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['class']
    });
  }

  private debounceTimer?: NodeJS.Timeout;
  private debounceLayoutCheck() {
    if (this.debounceTimer) {
      clearTimeout(this.debounceTimer);
    }
    
    this.debounceTimer = setTimeout(async () => {
      const healthReport = await dashboardHealthMonitor.runFullHealthCheck();
      if (healthReport.overallHealth === 'critical') {
        console.log('🚨 Critical layout issues detected - applying auto-repair...');
        await this.applyOptimalLayout();
      }
    }, 2000); // Wait 2 seconds after last change
  }

  public async applyLayout(layoutId: string): Promise<void> {
    const preset = this.layoutPresets.get(layoutId);
    if (!preset) {
      throw new Error(`Layout preset "${layoutId}" not found`);
    }

    console.log(`🎨 Applying layout: ${preset.name}`);
    
    // Trigger React re-render with new layout
    window.dispatchEvent(new CustomEvent('dashboard:layout:change', {
      detail: { 
        layoutId,
        config: preset.config
      }
    }));

    this.currentLayout = layoutId;
    
    // Store in localStorage
    localStorage.setItem('qt_auto_layout', layoutId);
    
    console.log(`✅ Layout "${preset.name}" applied successfully`);
  }

  public async applyOptimalLayout(): Promise<void> {
    await this.applyLayout('optimal');
  }

  public async resetToDefault(): Promise<void> {
    console.log('🔄 Resetting dashboard to default optimal layout...');
    await this.applyOptimalLayout();
    
    // Clear any corrupted state
    localStorage.removeItem('qt_dashboard_state');
    localStorage.removeItem('qt_layout_corrupted');
    
    // Force refresh components
    window.dispatchEvent(new CustomEvent('dashboard:reset:complete'));
  }

  // Specific repair methods
  private async repairDashboardContainer(): Promise<void> {
    console.log('🔧 Repairing dashboard container...');
    await this.applyOptimalLayout();
  }

  private async repairGridStructure(): Promise<void> {
    console.log('🔧 Repairing grid structure...');
    await this.applyOptimalLayout();
  }

  private async repairCandlesPosition(): Promise<void> {
    console.log('🔧 Moving Market Candles to correct position...');
    // Market Candles should be in position 11 (Row 4), NOT in header
    await this.applyOptimalLayout();
  }

  private async repairTradeHistoryWidth(): Promise<void> {
    console.log('🔧 Fixing Trade History width to full width...');
    // Trade History should span full width: xl:col-span-4
    await this.applyOptimalLayout();
  }

  private async repairPriceSync(): Promise<void> {
    console.log('🔧 Synchronizing prices across all components...');
    window.dispatchEvent(new CustomEvent('price:sync:force'));
  }

  private async repairApiConnection(): Promise<void> {
    console.log('🔧 Attempting to restore API connection...');
    window.dispatchEvent(new CustomEvent('api:reconnect'));
  }

  // Intelligent auto-detection and repair
  public async performIntelligentRepair(): Promise<void> {
    console.log('🤖 Starting intelligent dashboard repair...');
    
    // 1. Run health check
    const healthReport = await dashboardHealthMonitor.runFullHealthCheck();
    
    console.log(`📊 Health Report: ${healthReport.overallHealth.toUpperCase()}`);
    console.log(`📋 Found ${healthReport.issues.length} issues`);
    
    // 2. Log issues
    healthReport.issues.forEach(issue => {
      const emoji = issue.severity === 'critical' ? '🚨' : 
                   issue.severity === 'high' ? '⚠️' : 
                   issue.severity === 'medium' ? '🔶' : '💡';
      console.log(`${emoji} ${issue.description}`);
    });
    
    // 3. Apply recommendations
    if (healthReport.recommendations.length > 0) {
      console.log('💡 Recommendations:');
      healthReport.recommendations.forEach(rec => console.log(`   - ${rec}`));
    }
    
    // 4. Apply optimal layout if critical issues found
    if (healthReport.overallHealth === 'critical') {
      console.log('🎯 Applying optimal layout to resolve critical issues...');
      await this.applyOptimalLayout();
    }
    
    // 5. Re-check health after repair
    const postRepairHealth = await dashboardHealthMonitor.runFullHealthCheck();
    if (postRepairHealth.overallHealth === 'healthy') {
      console.log('🎉 Dashboard repair completed successfully!');
    } else {
      console.log('⚠️ Some issues remain after repair. Manual intervention may be needed.');
    }
  }

  // Public methods
  public enableAutoFix(): void {
    this.autoFixEnabled = true;
    console.log('✅ Auto-fix enabled');
  }

  public disableAutoFix(): void {
    this.autoFixEnabled = false;
    console.log('⏸️ Auto-fix disabled');
  }

  public getCurrentLayout(): string | undefined {
    return this.currentLayout;
  }

  public getAvailableLayouts(): LayoutPreset[] {
    return Array.from(this.layoutPresets.values());
  }

  public addCustomLayout(preset: LayoutPreset): void {
    this.layoutPresets.set(preset.id, preset);
  }
}

export const autoLayoutManager = AutoLayoutManager.getInstance();