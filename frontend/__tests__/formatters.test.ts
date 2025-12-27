import { describe, it, expect } from "vitest";
import {
  safeNum,
  safePercent,
  safeCurrency,
  safeInt,
  parseNumSafe,
} from "@/lib/formatters";

describe("safeNum()", () => {
  it("handles undefined values", () => {
    expect(safeNum(undefined)).toBe("0.00");
  });

  it("handles null values", () => {
    expect(safeNum(null as any)).toBe("0.00");
  });

  it("handles NaN", () => {
    expect(safeNum(NaN)).toBe("0.00");
  });

  it("handles Infinity", () => {
    expect(safeNum(Infinity)).toBe("0.00");
    expect(safeNum(-Infinity)).toBe("0.00");
  });

  it("formats valid numbers", () => {
    expect(safeNum(1.2345, 2)).toBe("1.23");
    expect(safeNum(1.2345, 3)).toBe("1.234"); // toFixed rounds down for trailing 5
    expect(safeNum(1.2345, 0)).toBe("1");
  });

  it("formats negative numbers", () => {
    expect(safeNum(-1.2345, 2)).toBe("-1.23");
  });

  it("formats zero correctly", () => {
    expect(safeNum(0, 2)).toBe("0.00");
    expect(safeNum(0, 4)).toBe("0.0000");
  });

  it("handles very small numbers", () => {
    expect(safeNum(0.0000001, 8)).toBe("0.00000010");
  });

  it("handles very large numbers", () => {
    expect(safeNum(1234567890.123, 2)).toBe("1234567890.12");
  });
});

describe("safePercent()", () => {
  it("adds percentage sign", () => {
    expect(safePercent(12.34)).toBe("12.34%");
  });

  it("handles undefined", () => {
    expect(safePercent(undefined)).toBe("0.00%");
  });

  it("handles negative percentages", () => {
    expect(safePercent(-5.67)).toBe("-5.67%");
  });
});

describe("safeCurrency()", () => {
  it("adds dollar sign by default", () => {
    expect(safeCurrency(1234.56)).toBe("$1234.56");
  });

  it("supports custom currency symbols", () => {
    expect(safeCurrency(1234.56, "€")).toBe("€1234.56");
    expect(safeCurrency(1234.56, "USDT ")).toBe("USDT 1234.56");
  });

  it("handles undefined", () => {
    expect(safeCurrency(undefined)).toBe("$0.00");
  });

  it("handles negative values", () => {
    expect(safeCurrency(-1234.56)).toBe("$-1234.56");
  });
});

describe("safeInt()", () => {
  it("formats integers without decimals", () => {
    expect(safeInt(123.456)).toBe("123");
  });

  it("handles undefined", () => {
    expect(safeInt(undefined)).toBe("0");
  });

  it("rounds to nearest integer", () => {
    expect(safeInt(123.6)).toBe("124");
    expect(safeInt(123.4)).toBe("123");
  });
});

describe("parseNumSafe()", () => {
  it("parses valid numbers", () => {
    expect(parseNumSafe(123.45)).toBe(123.45);
  });

  it("parses string numbers", () => {
    expect(parseNumSafe("123.45")).toBe(123.45);
  });

  it("returns 0 for invalid strings", () => {
    expect(parseNumSafe("abc")).toBe(0);
    expect(parseNumSafe("")).toBe(0);
  });

  it("returns 0 for undefined/null", () => {
    expect(parseNumSafe(undefined)).toBe(0);
    expect(parseNumSafe(null)).toBe(0);
  });

  it("returns 0 for NaN and Infinity", () => {
    expect(parseNumSafe(NaN)).toBe(0);
    expect(parseNumSafe(Infinity)).toBe(0);
  });

  it("handles objects", () => {
    expect(parseNumSafe({})).toBe(0);
    expect(parseNumSafe([])).toBe(0);
  });
});

describe("Real-world error scenarios", () => {
  it("prevents .toFixed errors on undefined", () => {
    const apiResponse: any = { confidence: undefined };
    expect(() => safeNum(apiResponse.confidence)).not.toThrow();
    expect(safeNum(apiResponse.confidence)).toBe("0.00");
  });

  it("prevents .toFixed errors on null", () => {
    const apiResponse: any = { pnl: null };
    expect(() => safeCurrency(apiResponse.pnl)).not.toThrow();
    expect(safeCurrency(apiResponse.pnl)).toBe("$0.00");
  });

  it("prevents .toFixed errors on division by zero", () => {
    const result = 10 / 0; // Infinity
    expect(() => safeNum(result)).not.toThrow();
    expect(safeNum(result)).toBe("0.00");
  });

  it("prevents .toFixed errors on NaN calculations", () => {
    const result = Math.sqrt(-1); // NaN
    expect(() => safePercent(result)).not.toThrow();
    expect(safePercent(result)).toBe("0.00%");
  });

  it("handles missing nested properties", () => {
    const data: any = {};
    expect(() => safeNum(data?.stats?.confidence)).not.toThrow();
    expect(safeNum(data?.stats?.confidence)).toBe("0.00");
  });
});
