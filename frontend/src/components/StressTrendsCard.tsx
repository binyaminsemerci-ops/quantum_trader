import { useEffect, useMemo, useState } from 'react';
import { fetchStressSummary } from '../api/stress';
import type { StressSummary, StressTask } from '../api/stress';

function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined) {
    return '-';
  }
  if (value < 60) {
    return `${value.toFixed(1)}s`;
  }
  const minutes = Math.floor(value / 60);
  const seconds = value - minutes * 60;
  if (minutes >= 60) {
    const hours = Math.floor(minutes / 60);
    const remainingMinutes = minutes % 60;
    return `${hours}h ${remainingMinutes}m ${seconds.toFixed(0)}s`;
  }
  return `${minutes}m ${seconds.toFixed(0)}s`;
}

function PassRateBar({ rate }: { rate: number }): JSX.Element {
  const clamped = Math.max(0, Math.min(rate, 100));
  return (
    <div className="flex items-center gap-2">
      <div className="relative h-2 flex-1 rounded bg-slate-800">
        <div
          className="absolute left-0 top-0 h-2 rounded bg-emerald-500"
          style={{ width: `${clamped}%` }}
        />
      </div>
      <span className="w-14 text-right tabular-nums text-sm text-slate-200">
        {clamped.toFixed(1)}%
      </span>
    </div>
  );
}

function TrendSparkline({ values }: { values: number[] }): JSX.Element {
  if (!values.length) {
    return <span className="text-xs text-slate-500">-</span>;
  }
  const width = 96;
  const height = 28;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = values.length === 1 ? 0 : (i / (values.length - 1)) * width;
      const y = height - ((v - min) / span) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline
        fill="none"
        stroke="#22c55e"
        strokeWidth={2}
        points={points}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function DurationSparkline({ values }: { values: number[] }): JSX.Element | null {
  if (!values.length) {
    return null;
  }
  const width = 240;
  const height = 56;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const points = values
    .map((v, i) => {
      const x = values.length === 1 ? 0 : (i / (values.length - 1)) * width;
      const y = height - ((v - min) / span) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join(' ');
  return (
    <svg width={width} height={height} viewBox={`0 0 ${width} ${height}`}>
      <polyline
        fill="none"
        stroke="#3b82f6"
        strokeWidth={2}
        points={points}
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function TaskTable({ tasks }: { tasks: StressTask[] }): JSX.Element {
  if (!tasks.length) {
    return <p className="text-sm text-slate-400">No task data yet.</p>;
  }
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead className="text-left text-xs uppercase tracking-wide text-slate-400">
          <tr>
            <th className="py-2 pr-4">Task</th>
            <th className="py-2 pr-4">Pass rate</th>
            <th className="py-2 pr-4">OK / Fail / Error / Skip</th>
            <th className="py-2">Trend</th>
          </tr>
        </thead>
        <tbody>
          {tasks.map((task) => (
            <tr key={task.name} className="border-t border-slate-800">
              <td className="py-2 pr-4 font-semibold text-slate-100">{task.name}</td>
              <td className="py-2 pr-4">
                <PassRateBar rate={task.pass_rate} />
              </td>
              <td className="py-2 pr-4 text-slate-200">
                <span className="tabular-nums">{task.counts.ok}</span>
                <span className="mx-1 text-slate-500">/</span>
                <span className="tabular-nums text-rose-400">{task.counts.fail}</span>
                <span className="mx-1 text-slate-500">/</span>
                <span className="tabular-nums text-amber-400">{task.counts.error}</span>
                <span className="mx-1 text-slate-500">/</span>
                <span className="tabular-nums text-slate-400">{task.counts.skipped}</span>
              </td>
              <td className="py-2">
                <TrendSparkline values={task.trend} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default function StressTrendsCard(): JSX.Element {
  const [summary, setSummary] = useState<StressSummary | null>(null);
  const [loading, setLoading] = useState<boolean>(false);
  const [loaded, setLoaded] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        setLoading(true);
        const payload = await fetchStressSummary();
        if (!cancelled) {
          setSummary(payload);
          setError(null);
          setLoaded(true);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : String(err));
          setLoaded(true);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }
    load();
    const timer = window.setInterval(load, 60000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, []);

  const durationSeries = summary?.duration_series ?? [];
  const lastRun = useMemo(() => {
    if (!summary?.recent_runs?.length) {
      return null;
    }
    return summary.recent_runs[summary.recent_runs.length - 1];
  }, [summary]);

  const lastRunStatus = useMemo(() => {
    if (!lastRun?.summary) {
      return '-';
    }
    const values = Object.values(lastRun.summary as Record<string, unknown>).filter(
      (value) => value !== undefined && value !== null,
    );
    if (!values.length) {
      return '-';
    }
    const allOk = values.every((value) => {
      if (value === 0 || value === '0') {
        return true;
      }
      if (value === 'skipped' || value === 'SKIPPED') {
        return true;
      }
      return false;
    });
    return allOk ? 'OK' : 'Needs attention';
  }, [lastRun]);

  const sortedTasks = useMemo<StressTask[]>(() => {
    if (!summary?.tasks?.length) {
      return [];
    }
    return [...summary.tasks].sort((a, b) => a.name.localeCompare(b.name));
  }, [summary]);

  return (
    <section className="rounded-lg border border-slate-800 bg-slate-900 p-4 text-slate-100 shadow-sm">
      <header className="mb-4 flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h3 className="text-lg font-semibold">Stress trends</h3>
          <p className="text-sm text-slate-400">
            Pass rate and duration pulled from artifacts/stress/aggregated.json
          </p>
          {summary?.source && (
            <p className="text-xs text-slate-500 mt-1">Source: {summary.source}</p>
          )}
        </div>
        {summary?.finished_at && (
          <span className="text-xs text-slate-400">
            Last updated {new Date(summary.finished_at).toLocaleString()}
          </span>
        )}
      </header>

      {!loaded && loading && <p className="text-sm text-slate-400">Loading stress metrics...</p>}
      {loaded && error && (
        <p className="text-sm text-rose-400">Failed to load stress metrics: {error}</p>
      )}
      {loaded && !error && summary === null && (
        <p className="text-sm text-slate-400">No stress runs yet.</p>
      )}

      {summary && !error && (
        <div className="space-y-6">
          <div className="grid gap-4 sm:grid-cols-3">
            <div className="rounded-md border border-slate-800 bg-slate-950/60 p-3">
              <p className="text-xs uppercase text-slate-500">Iterations</p>
              <p className="text-2xl font-semibold text-slate-100 tabular-nums">
                {summary.iterations}
              </p>
            </div>
            <div className="rounded-md border border-slate-800 bg-slate-950/60 p-3">
              <p className="text-xs uppercase text-slate-500">Avg duration</p>
              <p className="text-2xl font-semibold text-slate-100">
                {formatSeconds(summary.duration?.avg)}
              </p>
            </div>
            <div className="rounded-md border border-slate-800 bg-slate-950/60 p-3">
              <p className="text-xs uppercase text-slate-500">Last run</p>
              <p className="text-2xl font-semibold text-slate-100">{lastRunStatus}</p>
            </div>
          </div>

          <div>
            <h4 className="mb-2 text-sm font-semibold text-slate-300">Duration trend</h4>
            <DurationSparkline values={durationSeries} />
          </div>

          <TaskTable tasks={sortedTasks} />

          {summary.recent_runs?.length ? (
            <div>
              <h4 className="mb-2 text-sm font-semibold text-slate-300">Recent runs</h4>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead className="text-left text-xs uppercase tracking-wide text-slate-400">
                    <tr>
                      <th className="py-2 pr-4">Iteration</th>
                        <th className="py-2 pr-4">Time</th>
                      <th className="py-2 pr-4">Pytest</th>
                      <th className="py-2 pr-4">Backtest</th>
                      <th className="py-2 pr-4">Frontend</th>
                      <th className="py-2">Duration</th>
                    </tr>
                  </thead>
                  <tbody>
                    {summary.recent_runs.slice(-10).map((run, index) => (
                      <tr key={run.iteration ?? `row-${index}`} className="border-t border-slate-800">
                        <td className="py-2 pr-4 tabular-nums text-slate-200">
                          {run.iteration ?? '-'}
                        </td>
                          <td className="py-2 pr-4 text-slate-400">
                            {run.ts ? new Date(run.ts).toLocaleString() : '-'}
                          </td>
                        <td className="py-2 pr-4 text-slate-200">
                          {(() => {
                            const v = (run.summary as Record<string, unknown> | null)?.pytest;
                            return v === undefined || v === null ? '-' : String(v);
                          })()}
                        </td>
                        <td className="py-2 pr-4 text-slate-200">
                          {(() => {
                            const v = (run.summary as Record<string, unknown> | null)?.backtest;
                            return v === undefined || v === null ? '-' : String(v);
                          })()}
                        </td>
                        <td className="py-2 pr-4 text-slate-200">
                          {(() => {
                            const v = (run.summary as Record<string, unknown> | null)?.frontend_tests;
                            return v === undefined || v === null ? '-' : String(v);
                          })()}
                        </td>
                        <td className="py-2 text-slate-200">
                          {formatSeconds(run.total_duration)}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}
