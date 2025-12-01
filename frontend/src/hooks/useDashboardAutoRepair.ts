/**
 * Dashboard Auto-Repair Hook - React integration for intelligent dashboard management
 * Automatisk reparasjon integrasjon i React komponenter
 */

import { useEffect, useState, useCallback } from 'react';
import { dashboardHealthMonitor, type DashboardHealthReport } from '../services/dashboardHealthMonitor';
import { autoLayoutManager } from '../services/autoLayoutManager';

interface AutoRepairConfig {
  enabled: boolean;
  checkInterval: number; // milliseconds
  criticalThreshold: number; // number of critical issues to trigger auto-repair
  showNotifications: boolean;
}

interface AutoRepairState {
  isHealthy: boolean;
  lastCheck?: Date;
  issuesFound: number;
  criticalIssues: number;
  autoRepairActive: boolean;
  healthReport?: DashboardHealthReport;
}

export function useDashboardAutoRepair(config: Partial<AutoRepairConfig> = {}) {
  const defaultConfig: AutoRepairConfig = {
    enabled: true,
    checkInterval: 30000, // Check every 30 seconds
    criticalThreshold: 1, // Auto-repair if 1+ critical issues
    showNotifications: true,
    ...config
  };

  const [state, setState] = useState<AutoRepairState>({
    isHealthy: true,
    issuesFound: 0,
    criticalIssues: 0,
    autoRepairActive: false
  });

  const [repairLog, setRepairLog] = useState<string[]>([]);

  // Perform health check
  const performHealthCheck = useCallback(async () => {
    try {
      const healthReport = await dashboardHealthMonitor.runFullHealthCheck();
      
      const criticalIssues = healthReport.issues.filter(i => i.severity === 'critical').length;
      const isHealthy = healthReport.overallHealth === 'healthy';

      setState(prev => ({
        ...prev,
        isHealthy,
        lastCheck: new Date(),
        issuesFound: healthReport.issues.length,
        criticalIssues,
        healthReport
      }));

      // Auto-repair if critical issues exceed threshold
      if (defaultConfig.enabled && criticalIssues >= defaultConfig.criticalThreshold) {
        await triggerAutoRepair();
      }

      return healthReport;
    } catch (error) {
      console.error('Health check failed:', error);
      return null;
    }
  }, [defaultConfig.enabled, defaultConfig.criticalThreshold]);

  // Trigger auto-repair
  const triggerAutoRepair = useCallback(async () => {
    setState(prev => ({ ...prev, autoRepairActive: true }));
    
    const repairStartTime = new Date();
    const logEntry = `🚀 Auto-repair started at ${repairStartTime.toLocaleTimeString()}`;
    
    setRepairLog(prev => [...prev, logEntry]);

    try {
      await autoLayoutManager.performIntelligentRepair();
      
      const successEntry = `✅ Auto-repair completed at ${new Date().toLocaleTimeString()}`;
      setRepairLog(prev => [...prev, successEntry]);
      
      // Re-check health after repair
      setTimeout(() => {
        performHealthCheck();
      }, 2000);
      
    } catch (error) {
      const errorEntry = `❌ Auto-repair failed: ${error}`;
      setRepairLog(prev => [...prev, errorEntry]);
    } finally {
      setState(prev => ({ ...prev, autoRepairActive: false }));
    }
  }, [performHealthCheck]);

  // Manual repair trigger
  const manualRepair = useCallback(async () => {
    const logEntry = `🔧 Manual repair triggered at ${new Date().toLocaleTimeString()}`;
    setRepairLog(prev => [...prev, logEntry]);
    
    await triggerAutoRepair();
  }, [triggerAutoRepair]);

  // Reset to optimal layout
  const resetToOptimal = useCallback(async () => {
    const logEntry = `🎯 Reset to optimal layout at ${new Date().toLocaleTimeString()}`;
    setRepairLog(prev => [...prev, logEntry]);
    
    setState(prev => ({ ...prev, autoRepairActive: true }));
    
    try {
      await autoLayoutManager.resetToDefault();
      
      const successEntry = `🎉 Layout reset completed successfully`;
      setRepairLog(prev => [...prev, successEntry]);
      
      // Re-check health
      setTimeout(() => {
        performHealthCheck();
      }, 1000);
      
    } catch (error) {
      const errorEntry = `❌ Layout reset failed: ${error}`;
      setRepairLog(prev => [...prev, errorEntry]);
    } finally {
      setState(prev => ({ ...prev, autoRepairActive: false }));
    }
  }, [performHealthCheck]);

  // Clear repair log
  const clearRepairLog = useCallback(() => {
    setRepairLog([]);
  }, []);

  // Toggle auto-repair
  const toggleAutoRepair = useCallback((enabled: boolean) => {
    if (enabled) {
      dashboardHealthMonitor.enableAutoRepair();
      autoLayoutManager.enableAutoFix();
    } else {
      dashboardHealthMonitor.disableAutoRepair();
      autoLayoutManager.disableAutoFix();
    }
  }, []);

  // Setup periodic health checks
  useEffect(() => {
    if (!defaultConfig.enabled) return;

    const interval = setInterval(() => {
      performHealthCheck();
    }, defaultConfig.checkInterval);

    // Initial health check
    performHealthCheck();

    return () => clearInterval(interval);
  }, [defaultConfig.enabled, defaultConfig.checkInterval, performHealthCheck]);

  // Listen for layout events
  useEffect(() => {
    const handleLayoutChange = (event: CustomEvent) => {
      const logEntry = `🎨 Layout changed to: ${event.detail.layoutId}`;
      setRepairLog(prev => [...prev, logEntry]);
    };

    const handleResetComplete = () => {
      const logEntry = `🔄 Dashboard reset completed`;
      setRepairLog(prev => [...prev, logEntry]);
      performHealthCheck();
    };

    window.addEventListener('dashboard:layout:change', handleLayoutChange as EventListener);
    window.addEventListener('dashboard:reset:complete', handleResetComplete);

    return () => {
      window.removeEventListener('dashboard:layout:change', handleLayoutChange as EventListener);
      window.removeEventListener('dashboard:reset:complete', handleResetComplete);
    };
  }, [performHealthCheck]);

  return {
    // State
    isHealthy: state.isHealthy,
    issuesFound: state.issuesFound,
    criticalIssues: state.criticalIssues,
    autoRepairActive: state.autoRepairActive,
    lastCheck: state.lastCheck,
    healthReport: state.healthReport,
    repairLog,

    // Actions
    performHealthCheck,
    manualRepair,
    resetToOptimal,
    clearRepairLog,
    toggleAutoRepair,

    // Config
    config: defaultConfig
  };
}