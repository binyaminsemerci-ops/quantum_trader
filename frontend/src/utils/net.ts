// Generic network helper with absolute + relative fallback and basic timing
export interface FetchResult<T=any> { ok: boolean; data?: T; error?: string; durationMs: number; urlTried: string[] }

export function getBackendBase(): string {
  const base = (import.meta as any).env?.VITE_API_BASE_URL || 'http://localhost:8000';
  return base.replace(/\/$/, '');
}

export async function fetchWithFallback<T=any>(path: string, opts: RequestInit = {}): Promise<FetchResult<T>> {
  const trimmed = getBackendBase();
  const attempts: string[] = [];
  const candidates = path.startsWith('http') ? [path] : [ `${trimmed}${path.startsWith('/') ? '' : '/'}${path}`, path ];
  const start = performance.now();
  for (const url of candidates) {
    attempts.push(url);
    try {
      const res = await fetch(url, { ...opts, headers: { 'Accept': 'application/json', ...(opts.headers||{}) } });
      if (!res.ok) continue; // try next
      const json = await res.json();
      return { ok: true, data: json as T, durationMs: performance.now() - start, urlTried: attempts };
    } catch (e:any) {
      // try next
    }
  }
  return { ok: false, error: 'All fetch attempts failed', durationMs: performance.now() - start, urlTried: attempts };
}
