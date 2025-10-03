import { useEffect, useRef } from 'react';
import { useDashboardStore } from '../stores/dashboardStore';

export function useDebouncedLayoutSync(ms: number, enabled: boolean, token?: string) {
  const exportLayout = useDashboardStore(s => s.exportLayout);
  const layout = useDashboardStore(s => s.layout);
  const timer = useRef<number | null>(null);

  useEffect(() => {
    if (!enabled) return;
    if (timer.current) window.clearTimeout(timer.current);
    timer.current = window.setTimeout(async () => {
      try {
        const payload = JSON.parse(exportLayout());
        await fetch('/api/layout', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json', ...(token ? { Authorization: `Bearer ${token}` } : {}) },
          body: JSON.stringify({ layout: payload.layout, version: payload.layout.version, _schemaVersion: payload._schemaVersion })
        });
      } catch { /* silent */ }
    }, ms);
    return () => { if (timer.current) window.clearTimeout(timer.current); };
  }, [layout, ms, enabled, exportLayout, token]);
}
