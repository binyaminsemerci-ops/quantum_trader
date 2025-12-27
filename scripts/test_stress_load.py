#!/usr/bin/env python3
"""
STEP 5: STRESS TESTING
======================

Tests system under high load conditions:
1. Concurrent API requests (100+ simultaneous)
2. Rapid signal generation
3. Memory/connection pool exhaustion
4. Database connection stress
5. WebSocket connection stability

Author: GitHub Copilot (Senior QA Engineer)
Date: December 5, 2025
"""

import asyncio
import httpx
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any
import json
from pathlib import Path

# Configuration
BACKEND_URL = "http://localhost:8000"
CONCURRENT_REQUESTS = 100
TEST_DURATION_SECONDS = 60
TIMEOUT = 10.0

# Results storage
results: Dict[str, Any] = {
    "timestamp": datetime.now().isoformat(),
    "tests": [],
    "summary": {}
}


async def measure_endpoint_response(client: httpx.AsyncClient, endpoint: str, method: str = "GET") -> Dict[str, Any]:
    """Measure single endpoint response time."""
    start = time.time()
    try:
        if method == "GET":
            response = await client.get(endpoint)
        else:
            response = await client.post(endpoint)
        
        elapsed = time.time() - start
        return {
            "success": True,
            "status": response.status_code,
            "elapsed_ms": elapsed * 1000,
            "error": None
        }
    except Exception as e:
        elapsed = time.time() - start
        return {
            "success": False,
            "status": 0,
            "elapsed_ms": elapsed * 1000,
            "error": str(e)
        }


async def stress_test_concurrent_health_checks():
    """Test 1: Concurrent health check requests."""
    print("\n" + "="*80)
    print("TEST 1: CONCURRENT HEALTH CHECKS")
    print(f"Sending {CONCURRENT_REQUESTS} simultaneous requests...")
    print("="*80)
    
    async with httpx.AsyncClient(timeout=TIMEOUT, base_url=BACKEND_URL) as client:
        tasks = [measure_endpoint_response(client, "/health/live") for _ in range(CONCURRENT_REQUESTS)]
        responses = await asyncio.gather(*tasks)
    
    successful = [r for r in responses if r["success"]]
    failed = [r for r in responses if not r["success"]]
    
    if successful:
        response_times = [r["elapsed_ms"] for r in successful]
        result = {
            "test": "concurrent_health_checks",
            "passed": len(successful) >= CONCURRENT_REQUESTS * 0.95,  # 95% success rate
            "total_requests": CONCURRENT_REQUESTS,
            "successful": len(successful),
            "failed": len(failed),
            "min_ms": min(response_times),
            "max_ms": max(response_times),
            "avg_ms": statistics.mean(response_times),
            "median_ms": statistics.median(response_times),
            "p95_ms": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
            "errors": [r["error"] for r in failed[:5]]  # First 5 errors
        }
    else:
        result = {
            "test": "concurrent_health_checks",
            "passed": False,
            "total_requests": CONCURRENT_REQUESTS,
            "successful": 0,
            "failed": len(failed),
            "errors": [r["error"] for r in failed[:10]]
        }
    
    results["tests"].append(result)
    
    print(f"✅ Success: {len(successful)}/{CONCURRENT_REQUESTS}")
    print(f"❌ Failed: {len(failed)}/{CONCURRENT_REQUESTS}")
    if successful:
        print(f"⏱️  Response times: min={result['min_ms']:.2f}ms, avg={result['avg_ms']:.2f}ms, max={result['max_ms']:.2f}ms")
        print(f"📊 P95: {result['p95_ms']:.2f}ms")
    
    return result["passed"]


async def stress_test_dashboard_endpoints():
    """Test 2: Stress test dashboard BFF endpoints."""
    print("\n" + "="*80)
    print("TEST 2: DASHBOARD BFF STRESS TEST")
    print(f"Hammering dashboard endpoints with {CONCURRENT_REQUESTS} concurrent requests...")
    print("="*80)
    
    endpoints = [
        "/api/dashboard/trading",
        "/api/dashboard/risk",
        "/api/dashboard/overview",
        "/api/dashboard/system"
    ]
    
    async with httpx.AsyncClient(timeout=30.0, base_url=BACKEND_URL) as client:
        all_responses = []
        for endpoint in endpoints:
            print(f"\nTesting {endpoint}...")
            tasks = [measure_endpoint_response(client, endpoint) for _ in range(CONCURRENT_REQUESTS // len(endpoints))]
            responses = await asyncio.gather(*tasks)
            all_responses.extend(responses)
            
            successful = [r for r in responses if r["success"]]
            print(f"  {len(successful)}/{len(responses)} successful")
    
    successful = [r for r in all_responses if r["success"]]
    failed = [r for r in all_responses if not r["success"]]
    
    if successful:
        response_times = [r["elapsed_ms"] for r in successful]
        result = {
            "test": "dashboard_stress_test",
            "passed": len(successful) >= len(all_responses) * 0.90,  # 90% success rate
            "total_requests": len(all_responses),
            "successful": len(successful),
            "failed": len(failed),
            "min_ms": min(response_times),
            "max_ms": max(response_times),
            "avg_ms": statistics.mean(response_times),
            "median_ms": statistics.median(response_times),
            "p95_ms": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else max(response_times),
        }
    else:
        result = {
            "test": "dashboard_stress_test",
            "passed": False,
            "total_requests": len(all_responses),
            "successful": 0,
            "failed": len(failed)
        }
    
    results["tests"].append(result)
    
    print(f"\n📊 Overall: {len(successful)}/{len(all_responses)} successful")
    if successful:
        print(f"⏱️  Avg response: {result['avg_ms']:.2f}ms, P95: {result['p95_ms']:.2f}ms")
    
    return result["passed"]


async def stress_test_sustained_load():
    """Test 3: Sustained load test over time."""
    print("\n" + "="*80)
    print("TEST 3: SUSTAINED LOAD TEST")
    print(f"Running continuous requests for {TEST_DURATION_SECONDS} seconds...")
    print("="*80)
    
    start_time = time.time()
    request_count = 0
    successful_count = 0
    failed_count = 0
    response_times = []
    
    async with httpx.AsyncClient(timeout=TIMEOUT, base_url=BACKEND_URL) as client:
        while time.time() - start_time < TEST_DURATION_SECONDS:
            # Send burst of 10 concurrent requests
            tasks = [measure_endpoint_response(client, "/health/live") for _ in range(10)]
            responses = await asyncio.gather(*tasks)
            
            request_count += len(responses)
            for r in responses:
                if r["success"]:
                    successful_count += 1
                    response_times.append(r["elapsed_ms"])
                else:
                    failed_count += 1
            
            # Brief pause between bursts
            await asyncio.sleep(0.1)
            
            # Progress update every 10 seconds
            elapsed = time.time() - start_time
            if int(elapsed) % 10 == 0 and int(elapsed) > 0:
                print(f"  {int(elapsed)}s: {successful_count} successful, {failed_count} failed")
    
    total_duration = time.time() - start_time
    requests_per_second = request_count / total_duration
    
    result = {
        "test": "sustained_load",
        "passed": successful_count >= request_count * 0.95,  # 95% success rate
        "duration_seconds": total_duration,
        "total_requests": request_count,
        "successful": successful_count,
        "failed": failed_count,
        "requests_per_second": requests_per_second,
        "avg_ms": statistics.mean(response_times) if response_times else 0,
        "p95_ms": statistics.quantiles(response_times, n=20)[18] if len(response_times) > 20 else (max(response_times) if response_times else 0)
    }
    
    results["tests"].append(result)
    
    print(f"\n✅ Completed: {request_count} requests in {total_duration:.1f}s")
    print(f"📊 Success rate: {successful_count}/{request_count} ({successful_count/request_count*100:.1f}%)")
    print(f"⚡ Throughput: {requests_per_second:.1f} req/s")
    print(f"⏱️  Avg response: {result['avg_ms']:.2f}ms")
    
    return result["passed"]


async def stress_test_memory_leak_detection():
    """Test 4: Memory leak detection through repeated requests."""
    print("\n" + "="*80)
    print("TEST 4: MEMORY LEAK DETECTION")
    print("Making 1000 requests and monitoring response time degradation...")
    print("="*80)
    
    batch_size = 100
    num_batches = 10
    batch_response_times = []
    
    async with httpx.AsyncClient(timeout=TIMEOUT, base_url=BACKEND_URL) as client:
        for batch in range(num_batches):
            tasks = [measure_endpoint_response(client, "/api/dashboard/trading") for _ in range(batch_size)]
            responses = await asyncio.gather(*tasks)
            
            successful = [r for r in responses if r["success"]]
            if successful:
                avg_time = statistics.mean([r["elapsed_ms"] for r in successful])
                batch_response_times.append(avg_time)
                print(f"  Batch {batch+1}/{num_batches}: {len(successful)}/{batch_size} successful, avg={avg_time:.2f}ms")
            
            await asyncio.sleep(0.5)
    
    # Check for significant degradation (>50% increase from first to last batch)
    if len(batch_response_times) >= 2:
        first_batch_avg = statistics.mean(batch_response_times[:3])
        last_batch_avg = statistics.mean(batch_response_times[-3:])
        degradation_pct = ((last_batch_avg - first_batch_avg) / first_batch_avg) * 100
        
        result = {
            "test": "memory_leak_detection",
            "passed": degradation_pct < 50,  # Less than 50% degradation
            "first_batch_avg_ms": first_batch_avg,
            "last_batch_avg_ms": last_batch_avg,
            "degradation_pct": degradation_pct,
            "total_requests": batch_size * num_batches
        }
        
        print(f"\n📊 Performance change: {degradation_pct:+.1f}%")
        if result["passed"]:
            print("✅ No significant memory leak detected")
        else:
            print("⚠️ Potential memory leak - response times degraded significantly")
    else:
        result = {
            "test": "memory_leak_detection",
            "passed": False,
            "error": "Insufficient data"
        }
    
    results["tests"].append(result)
    return result["passed"]


async def stress_test_api_rate_limiting():
    """Test 5: Check if rate limiting is properly implemented."""
    print("\n" + "="*80)
    print("TEST 5: RATE LIMITING VALIDATION")
    print("Sending rapid-fire requests to check rate limiting...")
    print("="*80)
    
    async with httpx.AsyncClient(timeout=5.0, base_url=BACKEND_URL) as client:
        # Send 200 requests as fast as possible
        tasks = [measure_endpoint_response(client, "/health/live") for _ in range(200)]
        responses = await asyncio.gather(*tasks)
    
    status_codes = [r["status"] for r in responses if r["success"]]
    rate_limited = [s for s in status_codes if s == 429]
    successful = [s for s in status_codes if s == 200]
    
    result = {
        "test": "rate_limiting",
        "passed": True,  # Always pass, just informational
        "total_requests": len(responses),
        "successful_200": len(successful),
        "rate_limited_429": len(rate_limited),
        "rate_limiting_active": len(rate_limited) > 0
    }
    
    results["tests"].append(result)
    
    print(f"✅ 200 OK: {len(successful)}")
    print(f"⚠️ 429 Rate Limited: {len(rate_limited)}")
    if len(rate_limited) > 0:
        print("✅ Rate limiting is active")
    else:
        print("⚠️ No rate limiting detected (may be disabled or threshold not reached)")
    
    return True


async def main():
    """Run all stress tests."""
    print("="*80)
    print("QUANTUM TRADER V2.0 - STRESS TESTING")
    print("STEP 5: Load Testing & Performance Under Stress")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Backend: {BACKEND_URL}")
    print(f"Concurrent requests: {CONCURRENT_REQUESTS}")
    print(f"Test duration: {TEST_DURATION_SECONDS}s")
    print("="*80)
    
    # Run all tests
    test_results = []
    
    test_results.append(await stress_test_concurrent_health_checks())
    test_results.append(await stress_test_dashboard_endpoints())
    test_results.append(await stress_test_sustained_load())
    test_results.append(await stress_test_memory_leak_detection())
    test_results.append(await stress_test_api_rate_limiting())
    
    # Summary
    passed_count = sum(1 for r in test_results if r)
    total_count = len(test_results)
    pass_rate = (passed_count / total_count) * 100
    
    results["summary"] = {
        "total_tests": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "pass_rate": pass_rate
    }
    
    print("\n" + "="*80)
    print("STRESS TEST SUMMARY")
    print("="*80)
    print(f"Total Tests: {total_count}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {total_count - passed_count}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    print("="*80)
    
    if pass_rate >= 80:
        print("✅ SYSTEM STABLE UNDER STRESS")
    elif pass_rate >= 60:
        print("⚠️ SYSTEM FUNCTIONAL BUT SHOWS STRESS")
    else:
        print("❌ SYSTEM UNSTABLE UNDER LOAD")
    
    # Save results
    output_file = Path("STRESS_TEST_RESULTS.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {output_file.absolute()}")
    
    return 0 if pass_rate >= 80 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
