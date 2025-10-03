export type StressTaskCounts = {
  ok: number;
  fail: number;
  skipped: number;
  error: number;
};

export type StressTask = {
  name: string;
  counts: StressTaskCounts;
  pass_rate: number;
  trend: number[];
};

export type StressRun = {
  iteration: number | null;
  summary: Record<string, unknown> | null;
  total_duration: number | null;
  details?: string | null;
};

export type StressSummary = {
  status: string;
  source: string;
  started_at?: string | null;
  finished_at?: string | null;
  iterations: number;
  duration: {
    min: number | null;
    max: number | null;
    avg: number | null;
  };
  totals: {
    runs: number;
  };
  duration_series: number[];
  tasks: StressTask[];
  recent_runs: StressRun[];
};

export async function fetchStressSummary(): Promise<StressSummary | null> {
  const base = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';
  const res = await fetch(`${base}/stress/summary`, {
    headers: { Accept: 'application/json' },
  });
  if (res.status === 404) {
    return null;
  }
  if (!res.ok) {
    throw new Error(`Failed to load stress summary (HTTP ${res.status})`);
  }
  return (await res.json()) as StressSummary;
}
