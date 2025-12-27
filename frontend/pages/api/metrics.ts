/**
 * Prometheus Metrics Endpoint for Frontend
 * Exposes Next.js metrics in Prometheus format
 */

import type { NextApiRequest, NextApiResponse } from 'next';

interface MetricsData {
  httpRequestsTotal: number;
  httpRequestDuration: number;
  activeConnections: number;
  errorCount: number;
}

// In-memory metrics storage (in production, use Redis or dedicated metrics library)
let metrics: MetricsData = {
  httpRequestsTotal: 0,
  httpRequestDuration: 0,
  activeConnections: 0,
  errorCount: 0
};

export default function handler(req: NextApiRequest, res: NextApiResponse) {
  // Update request count
  metrics.httpRequestsTotal++;

  // Generate Prometheus metrics format
  const metricsOutput = `# HELP frontend_http_requests_total Total HTTP requests to frontend
# TYPE frontend_http_requests_total counter
frontend_http_requests_total ${metrics.httpRequestsTotal}

# HELP frontend_active_connections Active WebSocket connections
# TYPE frontend_active_connections gauge
frontend_active_connections ${metrics.activeConnections}

# HELP frontend_errors_total Total frontend errors
# TYPE frontend_errors_total counter
frontend_errors_total ${metrics.errorCount}

# HELP frontend_http_request_duration_seconds HTTP request duration
# TYPE frontend_http_request_duration_seconds gauge
frontend_http_request_duration_seconds ${metrics.httpRequestDuration}

# HELP frontend_uptime_seconds Frontend uptime in seconds
# TYPE frontend_uptime_seconds gauge
frontend_uptime_seconds ${Math.floor(process.uptime())}

# HELP nodejs_heap_size_used_bytes Node.js heap size used
# TYPE nodejs_heap_size_used_bytes gauge
nodejs_heap_size_used_bytes ${process.memoryUsage().heapUsed}

# HELP nodejs_heap_size_total_bytes Node.js heap size total
# TYPE nodejs_heap_size_total_bytes gauge
nodejs_heap_size_total_bytes ${process.memoryUsage().heapTotal}
`;

  res.setHeader('Content-Type', 'text/plain; charset=utf-8');
  res.status(200).send(metricsOutput);
}

// Helper function to update metrics (call from middleware)
export function updateMetrics(updates: Partial<MetricsData>) {
  metrics = { ...metrics, ...updates };
}
