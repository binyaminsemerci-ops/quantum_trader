// Utility functions for dashboard
import type { ESSState, ServiceStatus, MarketRegime } from './types';
import { safeNum, safePercent } from './formatters';

/**
 * Format number as currency
 */
export function formatCurrency(value: number): string {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value);
}

/**
 * Format percentage
 */
export function formatPercent(value: number, decimals: number = 2): string {
  const formatted = safeNum(value, decimals);
  return `${value >= 0 ? '+' : ''}${formatted}%`;
}

/**
 * Format large numbers with K/M suffix
 */
export function formatCompact(value: number): string {
  if (Math.abs(value) >= 1_000_000) {
    return `$${safeNum(value / 1_000_000, 2)}M`;
  }
  if (Math.abs(value) >= 1_000) {
    return `$${safeNum(value / 1_000, 1)}K`;
  }
  return formatCurrency(value);
}

/**
 * Get color class for PnL value
 */
export function getPnLColorClass(value: number): string {
  if (value > 0) return 'text-success';
  if (value < 0) return 'text-danger';
  return 'text-gray-500';
}

/**
 * Get background color class for PnL value
 */
export function getPnLBgClass(value: number): string {
  if (value > 0) return 'bg-success/10';
  if (value < 0) return 'bg-danger/10';
  return 'bg-gray-100';
}

/**
 * Get ESS state badge color
 */
export function getESSStateColor(state: ESSState): string {
  switch (state) {
    case 'ARMED':
      return 'bg-success text-white';
    case 'TRIPPED':
      return 'bg-danger text-white';
    case 'COOLING':
      return 'bg-warning text-white';
    case 'UNKNOWN':
    default:
      return 'bg-gray-500 text-white';
  }
}

/**
 * Get service status badge color
 */
export function getServiceStatusColor(status: ServiceStatus): string {
  switch (status) {
    case 'OK':
      return 'bg-success text-white';
    case 'DEGRADED':
      return 'bg-warning text-white';
    case 'DOWN':
      return 'bg-danger text-white';
    case 'UNKNOWN':
    default:
      return 'bg-gray-500 text-white';
  }
}

/**
 * Format timestamp as relative time
 */
export function formatRelativeTime(timestamp: string): string {
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);

  if (diffSecs < 60) return `${diffSecs}s ago`;
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  return date.toLocaleDateString();
}

/**
 * Format timestamp as time (HH:MM:SS)
 */
export function formatTime(timestamp: string): string {
  return new Date(timestamp).toLocaleTimeString('en-US', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

/**
 * Get regime badge color (Sprint 4 Del 2)
 */
export function getRegimeColor(regime: MarketRegime): string {
  switch (regime) {
    case 'HIGH_VOL_TRENDING':
      return 'text-danger';
    case 'LOW_VOL_TRENDING':
      return 'text-success';
    case 'HIGH_VOL_RANGING':
      return 'text-warning';
    case 'LOW_VOL_RANGING':
      return 'text-primary';
    case 'CHOPPY':
      return 'text-gray-600';
    case 'UNKNOWN':
    default:
      return 'text-gray-500';
  }
}

/**
 * Get regime badge background class (Sprint 4 Del 2)
 */
export function getRegimeBadgeClass(regime: MarketRegime): string {
  switch (regime) {
    case 'HIGH_VOL_TRENDING':
      return 'bg-danger text-white';
    case 'LOW_VOL_TRENDING':
      return 'bg-success text-white';
    case 'HIGH_VOL_RANGING':
      return 'bg-warning text-white';
    case 'LOW_VOL_RANGING':
      return 'bg-primary text-white';
    case 'CHOPPY':
      return 'bg-gray-600 text-white';
    case 'UNKNOWN':
    default:
      return 'bg-gray-500 text-white';
  }
}

/**
 * Get volatility bucket color (Sprint 4 Del 2)
 */
export function getVolatilityBucketColor(bucket: string): string {
  switch (bucket) {
    case 'LOW':
      return 'bg-success text-white';
    case 'MEDIUM':
      return 'bg-primary text-white';
    case 'HIGH':
      return 'bg-warning text-white';
    case 'EXTREME':
      return 'bg-danger text-white';
    default:
      return 'bg-gray-500 text-white';
  }
}
