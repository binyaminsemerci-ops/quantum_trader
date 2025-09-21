// Minimal reportWebVitals shim - matches CRA signature but is safe for this project.
// It accepts an optional onPerfEntry callback and forwards to console or sends
// it to an analytics endpoint. For now we'll keep it a no-op to avoid runtime
// side effects during migration.

type PerformanceEntryCallback = (entry: any) => void;

export default function reportWebVitals(onPerfEntry?: PerformanceEntryCallback): void {
  if (onPerfEntry && typeof onPerfEntry === 'function') {
    // If caller provides a handler, attempt to register it with the browser
    try {
      if (typeof window !== 'undefined' && (window as any).performance && (window as any).performance.getEntries) {
        const entries = (window as any).performance.getEntries();
        entries.forEach((e: any) => onPerfEntry(e));
      }
    } catch (e) {
      // swallow - this is optional instrumentation
    }
  }
}
