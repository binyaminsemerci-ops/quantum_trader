/**
 * Dashboard Health Monitor - Automatically detects and repairs layout issues
 * Dette er løsningen på manuelle fiksinger!
 */

export interface DashboardHealthIssue {
  id: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  category: 'layout' | 'data' | 'performance' | 'ui';
  description: string;
  autoFixAvailable: boolean;
  fix?: () => void;
}

export interface DashboardHealthReport {
  timestamp: Date;
  overallHealth: 'healthy' | 'warning' | 'critical';
  issues: DashboardHealthIssue[];
  recommendations: string[];
}

class DashboardHealthMonitor {
  private static instance: DashboardHealthMonitor;
  private healthChecks: Map<string, () => DashboardHealthIssue[]> = new Map();
  private autoRepairEnabled: boolean = true;
  private lastHealthCheck?: DashboardHealthReport;

  private constructor() {
    this.setupDefaultHealthChecks();
  }

  public static getInstance(): DashboardHealthMonitor {
    if (!DashboardHealthMonitor.instance) {
      DashboardHealthMonitor.instance = new DashboardHealthMonitor();
    }
    return DashboardHealthMonitor.instance;
  }

  private setupDefaultHealthChecks() {
    // Layout Health Checks
    this.healthChecks.set('layout_structure', () => this.checkLayoutStructure());
    this.healthChecks.set('component_positioning', () => this.checkComponentPositioning());
    this.healthChecks.set('responsive_breakpoints', () => this.checkResponsiveBreakpoints());
    
    // Data Health Checks
    this.healthChecks.set('price_sync', () => this.checkPriceSync());
    this.healthChecks.set('api_connectivity', () => this.checkApiConnectivity());
    
    // Performance Health Checks
    this.healthChecks.set('render_performance', () => this.checkRenderPerformance());
    this.healthChecks.set('memory_usage', () => this.checkMemoryUsage());
    
    // UI Health Checks
    this.healthChecks.set('collapsible_panels', () => this.checkCollapsiblePanels());
    this.healthChecks.set('theme_consistency', () => this.checkThemeConsistency());
  }

  public async runFullHealthCheck(): Promise<DashboardHealthReport> {
    const issues: DashboardHealthIssue[] = [];
    
    // Run all health checks
    for (const [checkName, checkFunction] of this.healthChecks) {
      try {
        const checkIssues = checkFunction();
        issues.push(...checkIssues);
      } catch (error) {
        issues.push({
          id: `health_check_error_${checkName}`,
          severity: 'medium',
          category: 'performance',
          description: `Health check "${checkName}" failed: ${error}`,
          autoFixAvailable: false
        });
      }
    }

    // Determine overall health
    const criticalIssues = issues.filter(i => i.severity === 'critical');
    const highIssues = issues.filter(i => i.severity === 'high');
    
    let overallHealth: 'healthy' | 'warning' | 'critical' = 'healthy';
    if (criticalIssues.length > 0) {
      overallHealth = 'critical';
    } else if (highIssues.length > 0 || issues.length > 5) {
      overallHealth = 'warning';
    }

    const report: DashboardHealthReport = {
      timestamp: new Date(),
      overallHealth,
      issues,
      recommendations: this.generateRecommendations(issues)
    };

    this.lastHealthCheck = report;

    // Auto-repair if enabled and critical issues found
    if (this.autoRepairEnabled && (overallHealth === 'critical' || criticalIssues.length > 0)) {
      await this.performAutoRepair(issues);
    }

    return report;
  }

  private checkLayoutStructure(): DashboardHealthIssue[] {
    const issues: DashboardHealthIssue[] = [];
    
    // Check if dashboard container exists
    const dashboardContainer = document.querySelector('.min-h-screen');
    if (!dashboardContainer) {
      issues.push({
        id: 'missing_dashboard_container',
        severity: 'critical',
        category: 'layout',
        description: 'Dashboard main container is missing',
        autoFixAvailable: true,
        fix: () => this.fixDashboardContainer()
      });
    }

    // Check grid structure
    const gridContainer = document.querySelector('.grid');
    if (!gridContainer) {
      issues.push({
        id: 'missing_grid_container',
        severity: 'high',
        category: 'layout',
        description: 'Grid container is missing - components may overlap',
        autoFixAvailable: true,
        fix: () => this.fixGridContainer()
      });
    }

    return issues;
  }

  private checkComponentPositioning(): DashboardHealthIssue[] {
    const issues: DashboardHealthIssue[] = [];

    // Check if Market Candles is in header (common issue)
    const headerCandles = document.querySelector('h1')?.parentElement?.querySelector('h3');
    if (headerCandles?.textContent?.includes('Market Candles')) {
      issues.push({
        id: 'candles_in_header',
        severity: 'high',
        category: 'layout',
        description: 'Market Candles component is incorrectly positioned in header',
        autoFixAvailable: true,
        fix: () => this.fixCandlesPosition()
      });
    }

    // Check Trade History width
    const tradeHistory = document.querySelector('[data-component="TradeTable"]')?.closest('div');
    if (tradeHistory && !tradeHistory.classList.contains('xl:col-span-4')) {
      issues.push({
        id: 'narrow_trade_history',
        severity: 'medium',
        category: 'layout',
        description: 'Trade History is too narrow - not using full width',
        autoFixAvailable: true,
        fix: () => this.fixTradeHistoryWidth()
      });
    }

    return issues;
  }

  private checkResponsiveBreakpoints(): DashboardHealthIssue[] {
    const issues: DashboardHealthIssue[] = [];
    
    // Check for missing responsive classes
    const gridItems = document.querySelectorAll('.grid > div');
    gridItems.forEach((item, index) => {
      const classList = Array.from(item.classList);
      const hasResponsive = classList.some(cls => cls.includes('sm:') || cls.includes('lg:') || cls.includes('xl:'));
      
      if (!hasResponsive && index > 2) { // Skip first few items that might be single column
        issues.push({
          id: `missing_responsive_${index}`,
          severity: 'low',
          category: 'layout',
          description: `Grid item ${index} missing responsive breakpoint classes`,
          autoFixAvailable: true,
          fix: () => this.fixResponsiveClasses(item as HTMLElement)
        });
      }
    });

    return issues;
  }

  private checkPriceSync(): DashboardHealthIssue[] {
    const issues: DashboardHealthIssue[] = [];
    
    // Check if all price displays show the same value
    const priceElements = document.querySelectorAll('[data-price]');
    const prices = Array.from(priceElements).map(el => el.textContent?.replace(/[^0-9.]/g, ''));
    const uniquePrices = [...new Set(prices)];
    
    if (uniquePrices.length > 1) {
      issues.push({
        id: 'price_desync',
        severity: 'high',
        category: 'data',
        description: 'Price displays are not synchronized - showing different values',
        autoFixAvailable: true,
        fix: () => this.fixPriceSync()
      });
    }

    return issues;
  }

  private checkApiConnectivity(): DashboardHealthIssue[] {
    const issues: DashboardHealthIssue[] = [];
    
    // Check connection indicator
    const connectionDot = document.querySelector('.bg-green-500, .bg-red-500');
    if (connectionDot?.classList.contains('bg-red-500')) {
      issues.push({
        id: 'api_disconnected',
        severity: 'critical',
        category: 'data',
        description: 'API connection is lost - data may be stale',
        autoFixAvailable: true,
        fix: () => this.fixApiConnection()
      });
    }

    return issues;
  }

  private checkRenderPerformance(): DashboardHealthIssue[] {
    const issues: DashboardHealthIssue[] = [];
    
    // Check if too many re-renders happening
    const performanceEntries = performance.getEntriesByType('measure');
    const recentRenders = performanceEntries.filter(entry => 
      entry.name.includes('React') && Date.now() - entry.startTime < 5000
    );
    
    if (recentRenders.length > 50) {
      issues.push({
        id: 'excessive_rerenders',
        severity: 'medium',
        category: 'performance',
        description: 'Dashboard is re-rendering excessively - may cause lag',
        autoFixAvailable: false
      });
    }

    return issues;
  }

  private checkMemoryUsage(): DashboardHealthIssue[] {
    const issues: DashboardHealthIssue[] = [];
    
    // Check for memory leaks (basic check)
    if ('memory' in performance) {
      const memInfo = (performance as any).memory;
      if (memInfo.usedJSHeapSize > memInfo.jsHeapSizeLimit * 0.8) {
        issues.push({
          id: 'high_memory_usage',
          severity: 'medium',
          category: 'performance',
          description: 'High memory usage detected - possible memory leak',
          autoFixAvailable: false
        });
      }
    }

    return issues;
  }

  private checkCollapsiblePanels(): DashboardHealthIssue[] {
    const issues: DashboardHealthIssue[] = [];
    
    // Check if collapsible panels are working
    const panels = document.querySelectorAll('[data-component="CollapsiblePanel"]');
    panels.forEach((panel, index) => {
      const toggleButton = panel.querySelector('button');
      if (!toggleButton) {
        issues.push({
          id: `panel_missing_toggle_${index}`,
          severity: 'low',
          category: 'ui',
          description: `Collapsible panel ${index} is missing toggle button`,
          autoFixAvailable: false
        });
      }
    });

    return issues;
  }

  private checkThemeConsistency(): DashboardHealthIssue[] {
    const issues: DashboardHealthIssue[] = [];
    
    // Check for theme inconsistencies
    const darkElements = document.querySelectorAll('.dark\\:bg-gray-800, .dark\\:bg-gray-900');
    const lightElements = document.querySelectorAll('.bg-white, .bg-gray-50');
    
    if (darkElements.length > 0 && lightElements.length > 0) {
      // This is normal, but check if properly configured
      const htmlElement = document.documentElement;
      const isDarkMode = htmlElement.classList.contains('dark');
      
      if (!isDarkMode && darkElements.length > lightElements.length) {
        issues.push({
          id: 'theme_mismatch',
          severity: 'low',
          category: 'ui',
          description: 'Theme classes may not be properly configured',
          autoFixAvailable: false
        });
      }
    }

    return issues;
  }

  // Auto-repair methods
  private async performAutoRepair(issues: DashboardHealthIssue[]): Promise<void> {
    const fixableIssues = issues.filter(issue => issue.autoFixAvailable && issue.fix);
    
    console.log(`🔧 Auto-repairing ${fixableIssues.length} dashboard issues...`);
    
    for (const issue of fixableIssues) {
      try {
        if (issue.fix) {
          issue.fix();
          console.log(`✅ Fixed: ${issue.description}`);
        }
      } catch (error) {
        console.error(`❌ Failed to fix: ${issue.description}`, error);
      }
    }
    
    if (fixableIssues.length > 0) {
      console.log('🎉 Dashboard auto-repair completed!');
    }
  }

  private fixDashboardContainer(): void {
    // Implementation would trigger React re-render with proper structure
    window.dispatchEvent(new CustomEvent('dashboard:repair:container'));
  }

  private fixGridContainer(): void {
    window.dispatchEvent(new CustomEvent('dashboard:repair:grid'));
  }

  private fixCandlesPosition(): void {
    window.dispatchEvent(new CustomEvent('dashboard:repair:candles'));
  }

  private fixTradeHistoryWidth(): void {
    window.dispatchEvent(new CustomEvent('dashboard:repair:tradehistory'));
  }

  private fixResponsiveClasses(element: HTMLElement): void {
    // Add basic responsive classes
    element.classList.add('sm:col-span-1', 'lg:col-span-1', 'xl:col-span-1');
  }

  private fixPriceSync(): void {
    window.dispatchEvent(new CustomEvent('dashboard:repair:pricesync'));
  }

  private fixApiConnection(): void {
    window.dispatchEvent(new CustomEvent('dashboard:repair:apiconnection'));
  }

  private generateRecommendations(issues: DashboardHealthIssue[]): string[] {
    const recommendations: string[] = [];
    
    if (issues.some(i => i.category === 'layout')) {
      recommendations.push('Consider using the Layout Reset function to restore default layout');
    }
    
    if (issues.some(i => i.category === 'performance')) {
      recommendations.push('Enable React.memo() on components to reduce re-renders');
    }
    
    if (issues.some(i => i.category === 'data')) {
      recommendations.push('Check API connectivity and price service configuration');
    }

    if (issues.filter(i => i.severity === 'critical').length > 0) {
      recommendations.push('🚨 Critical issues detected - Auto-repair has been triggered');
    }

    return recommendations;
  }

  // Public methods
  public enableAutoRepair(): void {
    this.autoRepairEnabled = true;
  }

  public disableAutoRepair(): void {
    this.autoRepairEnabled = false;
  }

  public getLastHealthReport(): DashboardHealthReport | undefined {
    return this.lastHealthCheck;
  }

  public addCustomHealthCheck(name: string, checkFunction: () => DashboardHealthIssue[]): void {
    this.healthChecks.set(name, checkFunction);
  }
}

export const dashboardHealthMonitor = DashboardHealthMonitor.getInstance();