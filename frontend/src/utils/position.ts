export function parseNumberInput(raw: string | number | undefined): number {
  if (raw == null) return NaN;
  if (typeof raw === 'number') return raw;
  const cleaned = String(raw).replace(/\s/g, '').replace(',', '.');
  return Number(cleaned);
}

export function calcPositionSize(balance: number, riskPercent: number, entry: string | number, stop: string | number): number | null {
  const e = parseNumberInput(entry);
  const s = parseNumberInput(stop);
  if (!isFinite(e) || !isFinite(s) || e === s) return null;
  const riskPerTrade = (riskPercent / 100) * balance;
  const riskPerUnit = Math.abs(e - s);
  const size = riskPerTrade / riskPerUnit;
  if (!isFinite(size) || size <= 0) return null;
  return size;
}
