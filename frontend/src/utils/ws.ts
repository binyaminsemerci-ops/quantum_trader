export type ToastShape = { message?: string; type?: string } | null;

export function safeParse(payloadStr: unknown): unknown {
  if (typeof payloadStr !== 'string') return undefined;
  try {
    return JSON.parse(payloadStr);
  } catch {
    return undefined;
  }
}

export function extractToastFromPayload(payload: unknown): ToastShape {
  if (!payload || typeof payload !== 'object') return null;
  const p = payload as Record<string, any>;
  if (Array.isArray(p.logs) && p.logs.length > 0) {
    const latest = p.logs[0];
    const msg = `Trade ${String(latest.status).toUpperCase()}: ${latest.symbol} ${latest.side} ${latest.qty}@${latest.price}`;
    return { message: msg, type: latest.status === 'accepted' ? 'success' : 'error' };
  }
  return null;
}

export default { safeParse, extractToastFromPayload };
