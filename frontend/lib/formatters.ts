/**
 * Safe numeric formatters to prevent .toFixed() errors
 * Handles undefined, null, NaN, and invalid values gracefully
 */

export function safeNum(
  value: number | undefined | null,
  decimals: number = 2
): string {
  // Handle invalid values
  if (
    value === undefined ||
    value === null ||
    isNaN(value) ||
    !isFinite(value)
  ) {
    if (decimals === 0) return "0";
    return "0." + "0".repeat(decimals);
  }

  try {
    return value.toFixed(decimals);
  } catch (error) {
    console.error("Error formatting number:", error);
    if (decimals === 0) return "0";
    return "0." + "0".repeat(decimals);
  }
}

export function safePercent(
  value: number | undefined | null,
  decimals: number = 2
): string {
  return safeNum(value, decimals) + "%";
}

export function safeCurrency(
  value: number | undefined | null,
  currency: string = "$",
  decimals: number = 2
): string {
  return currency + safeNum(value, decimals);
}

export function safeInt(value: number | undefined | null): string {
  return safeNum(value, 0);
}

/**
 * Safely parse a numeric value from unknown input
 */
export function parseNumSafe(value: unknown): number {
  if (typeof value === "number") {
    if (isNaN(value) || !isFinite(value)) return 0;
    return value;
  }

  if (typeof value === "string") {
    const parsed = parseFloat(value);
    if (isNaN(parsed) || !isFinite(parsed)) return 0;
    return parsed;
  }

  return 0;
}
