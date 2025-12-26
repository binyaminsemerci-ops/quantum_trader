/**
 * Safe Number Formatter
 * Prevents toFixed() errors on undefined/null/NaN values
 */
export function safeNum(value?: number | null, decimals = 2): string {
  if (typeof value !== "number" || isNaN(value) || !isFinite(value)) {
    return "0.00";
  }
  return value.toFixed(decimals);
}

/**
 * Safe Percentage Formatter
 */
export function safePercent(value?: number | null, decimals = 2): string {
  if (typeof value !== "number" || isNaN(value) || !isFinite(value)) {
    return "0.00%";
  }
  return `${(value * 100).toFixed(decimals)}%`;
}

/**
 * Alias for safePercent
 */
export const safePct = safePercent;

/**
 * Safe Currency Formatter
 */
export function safeCurrency(value?: number | null, decimals = 2): string {
  if (typeof value !== "number" || isNaN(value) || !isFinite(value)) {
    return "$0.00";
  }
  return `$${value.toLocaleString(undefined, { 
    minimumFractionDigits: decimals, 
    maximumFractionDigits: decimals 
  })}`;
}

/**
 * Safe Integer Formatter
 */
export function safeInt(value?: number | null): string {
  if (typeof value !== "number" || isNaN(value) || !isFinite(value)) {
    return "0";
  }
  return Math.round(value).toLocaleString();
}
