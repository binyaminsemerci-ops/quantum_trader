"""
Prometheus Metrics Endpoint for QuantumFond Backend
Exposes FastAPI metrics in Prometheus format
"""

from fastapi import APIRouter, Response
from datetime import datetime
import psutil
import time

router = APIRouter()

# Metrics storage
start_time = time.time()
request_count = 0
error_count = 0


@router.get("/metrics", response_class=Response)
async def metrics():
    """
    Prometheus metrics endpoint
    Returns metrics in Prometheus text format
    """
    global request_count
    request_count += 1

    # Get system metrics
    process = psutil.Process()
    memory_info = process.memory_info()
    cpu_percent = process.cpu_percent(interval=0.1)
    
    # Calculate uptime
    uptime = time.time() - start_time

    # Generate Prometheus format
    metrics_text = f"""# HELP backend_http_requests_total Total HTTP requests to backend
# TYPE backend_http_requests_total counter
backend_http_requests_total {request_count}

# HELP backend_errors_total Total backend errors
# TYPE backend_errors_total counter
backend_errors_total {error_count}

# HELP backend_uptime_seconds Backend uptime in seconds
# TYPE backend_uptime_seconds gauge
backend_uptime_seconds {uptime:.2f}

# HELP backend_memory_rss_bytes Resident Set Size memory
# TYPE backend_memory_rss_bytes gauge
backend_memory_rss_bytes {memory_info.rss}

# HELP backend_memory_vms_bytes Virtual Memory Size
# TYPE backend_memory_vms_bytes gauge
backend_memory_vms_bytes {memory_info.vms}

# HELP backend_cpu_usage_percent CPU usage percentage
# TYPE backend_cpu_usage_percent gauge
backend_cpu_usage_percent {cpu_percent}

# HELP backend_open_fds Number of open file descriptors
# TYPE backend_open_fds gauge
backend_open_fds {process.num_fds()}

# HELP backend_threads_count Number of threads
# TYPE backend_threads_count gauge
backend_threads_count {process.num_threads()}
"""

    return Response(content=metrics_text, media_type="text/plain; charset=utf-8")


@router.get("/health")
async def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "uptime_seconds": time.time() - start_time,
        "timestamp": datetime.utcnow().isoformat()
    }


# Helper to increment error count
def increment_error_count():
    global error_count
    error_count += 1
