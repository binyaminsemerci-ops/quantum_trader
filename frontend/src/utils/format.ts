// Shared number formatting helpers
const defaultLocale = navigator?.language || 'en-US';

const cache: Record<string, Intl.NumberFormat> = {};
function getFormatter(key: string, opts: Intl.NumberFormatOptions) {
  if (!cache[key]) cache[key] = new Intl.NumberFormat(defaultLocale, opts);
  return cache[key];
}

export function formatNumber(value: number | undefined | null, opts: Intl.NumberFormatOptions = {}) {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return getFormatter(JSON.stringify(opts), opts).format(value);
}

export function formatCurrency(value: number | undefined | null, currency: string = 'USDC', fractionDigits = 2) {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  // For USDC, use custom formatting since it's not a standard currency
  if (currency === 'USDC') {
    return `${formatNumber(value, { minimumFractionDigits: fractionDigits, maximumFractionDigits: fractionDigits })} USDC`;
  }
  return getFormatter(`cur-${currency}-${fractionDigits}`, { style: 'currency', currency, minimumFractionDigits: fractionDigits, maximumFractionDigits: fractionDigits }).format(value);
}

export function formatPercent(value: number | undefined | null, fractionDigits = 2) {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return `${value > 0 ? '+' : ''}${formatNumber(value, { minimumFractionDigits: fractionDigits, maximumFractionDigits: fractionDigits })}%`;
}

export function formatCompact(value: number | undefined | null) {
  if (value === undefined || value === null || Number.isNaN(value)) return '—';
  return getFormatter('compact', { notation: 'compact', maximumFractionDigits: 2 }).format(value);
}
