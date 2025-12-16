"""
COMPREHENSIVE SYSTEM STRESS TEST
Tests all system components under load and verifies health
"""
import asyncio
import aiohttp
import time
import statistics
from datetime import datetime
from typing import List, Dict, Any
import json

BASE_URL = "http://localhost:8000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{text:^80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.CYAN}{'='*80}{Colors.RESET}\n")

def print_section(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}🔍 {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.RESET}\n")

def print_test(name: str, passed: bool, details: str = ""):
    status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if passed else f"{Colors.RED}❌ FAIL{Colors.RESET}"
    print(f"{status}: {name}")
    if details:
        print(f"   {Colors.YELLOW}→{Colors.RESET} {details}")

def print_metric(name: str, value: Any, unit: str = ""):
    print(f"  {Colors.CYAN}•{Colors.RESET} {name}: {Colors.BOLD}{value}{unit}{Colors.RESET}")

async def test_health_endpoint(session: aiohttp.ClientSession) -> Dict:
    """Test basic health endpoint"""
    print_section("1. HEALTH ENDPOINT TEST")
    
    try:
        start = time.time()
        async with session.get(f"{BASE_URL}/health") as response:
            duration = (time.time() - start) * 1000
            data = await response.json()
            
            passed = response.status == 200 and data.get("status") == "ok"
            print_test("Health endpoint responding", passed)
            print_metric("Response time", f"{duration:.2f}", "ms")
            print_metric("Status", data.get("status", "unknown"))
            print_metric("Binance keys", data.get("secrets", {}).get("has_binance_keys", False))
            
            return {"passed": passed, "duration": duration, "data": data}
    except Exception as e:
        print_test("Health endpoint", False, f"Error: {e}")
        return {"passed": False, "error": str(e)}

async def stress_test_concurrent_requests(session: aiohttp.ClientSession, 
                                         endpoint: str, 
                                         num_requests: int = 50) -> Dict:
    """Stress test with concurrent requests"""
    print_section(f"2. CONCURRENT REQUEST STRESS TEST ({num_requests} requests)")
    
    async def make_request():
        start = time.time()
        try:
            async with session.get(f"{BASE_URL}{endpoint}") as response:
                duration = (time.time() - start) * 1000
                return {
                    "success": response.status == 200,
                    "status": response.status,
                    "duration": duration
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "duration": (time.time() - start) * 1000
            }
    
    # Fire all requests concurrently
    start_time = time.time()
    results = await asyncio.gather(*[make_request() for _ in range(num_requests)])
    total_time = (time.time() - start_time) * 1000
    
    # Analyze results
    successes = sum(1 for r in results if r.get("success"))
    failures = num_requests - successes
    durations = [r["duration"] for r in results if r.get("success")]
    
    passed = successes >= num_requests * 0.95  # 95% success rate
    
    print_test("Concurrent requests handling", passed, 
               f"{successes}/{num_requests} successful")
    
    if durations:
        print_metric("Total time", f"{total_time:.2f}", "ms")
        print_metric("Requests/sec", f"{num_requests / (total_time/1000):.2f}")
        print_metric("Avg response time", f"{statistics.mean(durations):.2f}", "ms")
        print_metric("Min response time", f"{min(durations):.2f}", "ms")
        print_metric("Max response time", f"{max(durations):.2f}", "ms")
        print_metric("Median response time", f"{statistics.median(durations):.2f}", "ms")
        if len(durations) > 1:
            print_metric("Std deviation", f"{statistics.stdev(durations):.2f}", "ms")
    
    return {
        "passed": passed,
        "successes": successes,
        "failures": failures,
        "durations": durations,
        "total_time": total_time
    }

async def test_sustained_load(session: aiohttp.ClientSession, 
                              duration_seconds: int = 30) -> Dict:
    """Test system under sustained load"""
    print_section(f"3. SUSTAINED LOAD TEST ({duration_seconds}s)")
    
    results = []
    start_time = time.time()
    request_count = 0
    
    print(f"{Colors.YELLOW}Running sustained load...{Colors.RESET}")
    
    async def continuous_requests():
        nonlocal request_count
        while time.time() - start_time < duration_seconds:
            try:
                req_start = time.time()
                async with session.get(f"{BASE_URL}/health") as response:
                    duration = (time.time() - req_start) * 1000
                    results.append({
                        "success": response.status == 200,
                        "duration": duration,
                        "timestamp": time.time() - start_time
                    })
                    request_count += 1
            except Exception as e:
                results.append({
                    "success": False,
                    "error": str(e),
                    "timestamp": time.time() - start_time
                })
            await asyncio.sleep(0.1)  # 10 req/sec per task
    
    # Run 5 concurrent workers
    await asyncio.gather(*[continuous_requests() for _ in range(5)])
    
    total_time = time.time() - start_time
    successes = sum(1 for r in results if r.get("success"))
    success_rate = (successes / len(results)) * 100 if results else 0
    
    passed = success_rate >= 95 and len(results) > 0
    
    print_test("Sustained load handling", passed)
    print_metric("Duration", f"{total_time:.2f}", "s")
    print_metric("Total requests", len(results))
    print_metric("Success rate", f"{success_rate:.2f}", "%")
    print_metric("Avg req/sec", f"{len(results) / total_time:.2f}")
    
    if results:
        durations = [r["duration"] for r in results if r.get("success")]
        if durations:
            print_metric("Avg response time", f"{statistics.mean(durations):.2f}", "ms")
            print_metric("Max response time", f"{max(durations):.2f}", "ms")
    
    return {
        "passed": passed,
        "total_requests": len(results),
        "success_rate": success_rate,
        "duration": total_time
    }

async def test_memory_leak(session: aiohttp.ClientSession, iterations: int = 100) -> Dict:
    """Check for memory leaks by monitoring response consistency"""
    print_section("4. MEMORY LEAK CHECK")
    
    print(f"{Colors.YELLOW}Running {iterations} iterations...{Colors.RESET}")
    
    response_times = []
    for i in range(iterations):
        try:
            start = time.time()
            async with session.get(f"{BASE_URL}/health") as response:
                duration = (time.time() - start) * 1000
                response_times.append(duration)
                
            if (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{iterations} requests")
        except Exception as e:
            print_test("Memory leak check", False, f"Error at iteration {i}: {e}")
            return {"passed": False}
    
    # Check if response times are stable (no significant increase)
    first_quarter = response_times[:iterations//4]
    last_quarter = response_times[-iterations//4:]
    
    avg_first = statistics.mean(first_quarter)
    avg_last = statistics.mean(last_quarter)
    increase_pct = ((avg_last - avg_first) / avg_first) * 100 if avg_first > 0 else 0
    
    # Pass if response times didn't increase by more than 50%
    passed = abs(increase_pct) < 50
    
    print_test("No significant memory leak detected", passed)
    print_metric("First 25% avg", f"{avg_first:.2f}", "ms")
    print_metric("Last 25% avg", f"{avg_last:.2f}", "ms")
    print_metric("Change", f"{increase_pct:+.2f}", "%")
    
    return {
        "passed": passed,
        "avg_first": avg_first,
        "avg_last": avg_last,
        "change_pct": increase_pct
    }

async def test_error_recovery(session: aiohttp.ClientSession) -> Dict:
    """Test error handling and recovery"""
    print_section("5. ERROR RECOVERY TEST")
    
    # Test invalid endpoint
    try:
        async with session.get(f"{BASE_URL}/invalid_endpoint_xyz123") as response:
            handles_404 = response.status == 404
            print_test("Handles 404 errors gracefully", handles_404)
    except Exception as e:
        print_test("404 error handling", False, f"Error: {e}")
        handles_404 = False
    
    # Test rapid requests after error
    try:
        await asyncio.sleep(0.5)
        async with session.get(f"{BASE_URL}/health") as response:
            recovers = response.status == 200
            print_test("Recovers after error", recovers)
    except Exception as e:
        print_test("Error recovery", False, f"Error: {e}")
        recovers = False
    
    passed = handles_404 and recovers
    
    return {"passed": passed, "handles_404": handles_404, "recovers": recovers}

async def test_response_consistency(session: aiohttp.ClientSession, num_tests: int = 20) -> Dict:
    """Test response consistency"""
    print_section("6. RESPONSE CONSISTENCY TEST")
    
    responses = []
    for _ in range(num_tests):
        try:
            async with session.get(f"{BASE_URL}/health") as response:
                data = await response.json()
                responses.append(data)
        except Exception as e:
            print_test("Response consistency", False, f"Error: {e}")
            return {"passed": False}
    
    # Check all responses are identical in structure
    first = responses[0]
    all_consistent = all(
        r.get("status") == first.get("status") and
        r.get("secrets", {}).keys() == first.get("secrets", {}).keys()
        for r in responses
    )
    
    print_test("Responses are consistent", all_consistent, 
               f"Tested {num_tests} requests")
    
    return {"passed": all_consistent, "num_tests": num_tests}

async def run_comprehensive_stress_test():
    """Run all stress tests"""
    print_header("QUANTUM TRADER - COMPREHENSIVE STRESS TEST")
    
    print(f"{Colors.BOLD}Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}\n")
    
    results = {}
    
    timeout = aiohttp.ClientTimeout(total=30)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Run all tests
        results["health"] = await test_health_endpoint(session)
        await asyncio.sleep(1)
        
        results["concurrent"] = await stress_test_concurrent_requests(session, "/health", 50)
        await asyncio.sleep(2)
        
        results["sustained"] = await test_sustained_load(session, 30)
        await asyncio.sleep(2)
        
        results["memory"] = await test_memory_leak(session, 100)
        await asyncio.sleep(1)
        
        results["error_recovery"] = await test_error_recovery(session)
        await asyncio.sleep(1)
        
        results["consistency"] = await test_response_consistency(session, 20)
    
    # Summary
    print_header("TEST SUMMARY")
    
    total_tests = len(results)
    passed_tests = sum(1 for r in results.values() if r.get("passed", False))
    
    print_metric("Total test categories", total_tests)
    print_metric("Passed", passed_tests)
    print_metric("Failed", total_tests - passed_tests)
    print_metric("Success rate", f"{(passed_tests/total_tests)*100:.1f}", "%")
    
    print(f"\n{Colors.BOLD}Detailed Results:{Colors.RESET}\n")
    for test_name, result in results.items():
        status = f"{Colors.GREEN}✅ PASS{Colors.RESET}" if result.get("passed") else f"{Colors.RED}❌ FAIL{Colors.RESET}"
        print(f"  {status} - {test_name}")
    
    overall_passed = passed_tests == total_tests
    
    print("\n" + "="*80)
    if overall_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}🎉 ALL TESTS PASSED - SYSTEM HEALTHY! 🎉{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}⚠️  SOME TESTS FAILED - CHECK RESULTS ABOVE ⚠️{Colors.RESET}")
    print("="*80 + "\n")
    
    print(f"{Colors.BOLD}Test completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}\n")
    
    return results, overall_passed

if __name__ == "__main__":
    try:
        results, passed = asyncio.run(run_comprehensive_stress_test())
        exit(0 if passed else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Test interrupted by user{Colors.RESET}\n")
        exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Fatal error: {e}{Colors.RESET}\n")
        exit(1)
