#!/usr/bin/env python3
"""
End-to-End System Integration Tests

Comprehensive integration tests for the complete Quantum Trader system including:
- Full stack integration (Frontend + Backend + AI)
- User workflow simulation
- Real-time data flow testing  
- Performance under load
- Error recovery testing
- Cross-component communication
"""

import asyncio
import json
import subprocess
import sys
import time
import websocket
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests

from test_backend_system import BackendSystemTester
from test_frontend_system import FrontendSystemTester

class E2ESystemTester:
    """End-to-end system integration testing."""
    
    def __init__(self):
        self.backend_tester = BackendSystemTester()
        self.frontend_tester = FrontendSystemTester()
        self.test_results: List[Dict[str, Any]] = []
        
        # URLs
        self.backend_url = "http://localhost:8000"
        self.frontend_url = "http://localhost:5173"
        self.ws_url = "ws://localhost:8000/ws/dashboard"
        
        # Processes
        self.backend_process: Optional[subprocess.Popen] = None
        self.frontend_process: Optional[subprocess.Popen] = None
        
    def start_full_stack(self) -> bool:
        """Start both backend and frontend servers."""
        print("🚀 Starting full stack for E2E testing...")
        
        # Start backend
        if not self.backend_tester.start_backend_server():
            print("❌ Failed to start backend")
            return False
        
        # Start frontend  
        if not self.frontend_tester.start_frontend_dev_server():
            print("❌ Failed to start frontend")
            return False
        
        print("✅ Full stack started successfully")
        return True
    
    def stop_full_stack(self):
        """Stop all servers."""
        print("🛑 Stopping full stack...")
        self.backend_tester.stop_backend_server()
        self.frontend_tester.stop_servers()
    
    def test_full_stack_communication(self) -> Dict[str, Any]:
        """Test communication between frontend and backend."""
        start_time = time.time()
        
        try:
            print("🔄 Testing full stack communication...")
            
            # Test 1: Frontend can reach backend
            frontend_to_backend = []
            api_endpoints = ["/health", "/api/prices", "/api/trades", "/api/portfolio"]
            
            for endpoint in api_endpoints:
                try:
                    # Simulate frontend making API call
                    response = requests.get(f"{self.backend_url}{endpoint}", 
                                          headers={"User-Agent": "Frontend-Test"}, 
                                          timeout=5)
                    frontend_to_backend.append({
                        "endpoint": endpoint,
                        "success": response.status_code == 200,
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds() * 1000
                    })
                except Exception as e:
                    frontend_to_backend.append({
                        "endpoint": endpoint,
                        "success": False,
                        "error": str(e)
                    })
            
            # Test 2: CORS headers for frontend
            cors_test = None
            try:
                response = requests.options(f"{self.backend_url}/api/prices",
                                          headers={
                                              "Origin": "http://localhost:5173",
                                              "Access-Control-Request-Method": "GET"
                                          })
                cors_test = {
                    "cors_enabled": "access-control-allow-origin" in response.headers,
                    "allows_frontend": response.headers.get("access-control-allow-origin") in ["*", "http://localhost:5173"],
                    "status_code": response.status_code
                }
            except Exception as e:
                cors_test = {"error": str(e)}
            
            # Calculate success metrics
            successful_endpoints = sum(1 for ep in frontend_to_backend if ep.get("success", False))
            success_rate = successful_endpoints / len(frontend_to_backend)
            
            result = {
                "test_name": "full_stack_communication",
                "success": success_rate >= 0.75 and cors_test.get("cors_enabled", False),
                "test_duration_ms": (time.time() - start_time) * 1000,
                "endpoints_tested": len(frontend_to_backend),
                "endpoints_successful": successful_endpoints,
                "success_rate": success_rate,
                "cors_test": cors_test,
                "api_results": frontend_to_backend,
                "error": None if success_rate >= 0.75 else f"Only {success_rate:.1%} endpoints working"
            }
            
        except Exception as e:
            result = {
                "test_name": "full_stack_communication", 
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000
            }
        
        self.test_results.append(result)
        return result
    
    def test_real_time_data_flow(self) -> Dict[str, Any]:
        """Test real-time WebSocket data flow."""
        start_time = time.time()
        messages_received = []
        connection_events = []
        
        def on_message(ws, message):
            try:
                data = json.loads(message)
                messages_received.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": data.get("type", "unknown"),
                    "size": len(message)
                })
            except:
                messages_received.append({
                    "timestamp": datetime.now().isoformat(),
                    "raw": message[:100]  # First 100 chars
                })
        
        def on_open(ws):
            connection_events.append("connected")
            # Send test messages to simulate frontend interactions
            test_messages = [
                {"type": "subscribe", "symbols": ["BTCUSDT", "ETHUSDT"]},
                {"type": "ping", "timestamp": datetime.now().isoformat()},
                {"type": "get_portfolio", "user_id": "test_user"}
            ]
            
            for msg in test_messages:
                ws.send(json.dumps(msg))
                time.sleep(0.5)
        
        def on_error(ws, error):
            connection_events.append(f"error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            connection_events.append("closed")
        
        try:
            print("🌐 Testing real-time WebSocket data flow...")
            
            ws = websocket.WebSocketApp(
                self.ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Run WebSocket in thread
            ws_thread = threading.Thread(target=ws.run_forever)
            ws_thread.daemon = True
            ws_thread.start()
            
            # Let it run for 10 seconds
            time.sleep(10)
            ws.close()
            
            # Wait for cleanup
            ws_thread.join(timeout=2)
            
            result = {
                "test_name": "real_time_data_flow",
                "success": len(messages_received) > 0 and "connected" in connection_events,
                "test_duration_ms": (time.time() - start_time) * 1000,
                "messages_received": len(messages_received),
                "connection_events": connection_events,
                "message_types": list(set(msg.get("type", "unknown") for msg in messages_received)),
                "sample_messages": messages_received[:5],
                "error": None if len(messages_received) > 0 else "No messages received"
            }
            
        except Exception as e:
            result = {
                "test_name": "real_time_data_flow",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000
            }
        
        self.test_results.append(result)
        return result
    
    def test_user_workflow_simulation(self) -> Dict[str, Any]:
        """Simulate complete user workflows."""
        start_time = time.time()
        
        try:
            print("👤 Testing user workflow simulation...")
            
            workflow_steps = []
            
            # Step 1: User opens frontend
            try:
                response = requests.get(self.frontend_url, timeout=5)
                workflow_steps.append({
                    "step": "load_frontend",
                    "success": response.status_code == 200,
                    "details": f"HTTP {response.status_code}"
                })
            except Exception as e:
                workflow_steps.append({
                    "step": "load_frontend",
                    "success": False,
                    "error": str(e)
                })
            
            # Step 2: Frontend loads initial data
            api_calls = [
                ("portfolio", "/api/portfolio"),
                ("prices", "/api/prices"),
                ("trades", "/api/trades?limit=10"),
                ("watchlist", "/api/watchlist")
            ]
            
            for name, endpoint in api_calls:
                try:
                    response = requests.get(f"{self.backend_url}{endpoint}", timeout=5)
                    workflow_steps.append({
                        "step": f"load_{name}",
                        "success": response.status_code == 200,
                        "details": f"HTTP {response.status_code}, {len(response.content)} bytes"
                    })
                except Exception as e:
                    workflow_steps.append({
                        "step": f"load_{name}",
                        "success": False,
                        "error": str(e)
                    })
            
            # Step 3: User requests chart data
            symbols = ["BTCUSDT", "ETHUSDT"]
            for symbol in symbols:
                try:
                    response = requests.get(f"{self.backend_url}/api/chart/{symbol}", timeout=10)
                    workflow_steps.append({
                        "step": f"load_chart_{symbol}",
                        "success": response.status_code == 200,
                        "details": f"HTTP {response.status_code}"
                    })
                except Exception as e:
                    workflow_steps.append({
                        "step": f"load_chart_{symbol}",
                        "success": False,
                        "error": str(e)
                    })
            
            # Step 4: User checks system stats
            try:
                response = requests.get(f"{self.backend_url}/api/stats", timeout=5)
                workflow_steps.append({
                    "step": "load_stats",
                    "success": response.status_code == 200,
                    "details": f"HTTP {response.status_code}"
                })
            except Exception as e:
                workflow_steps.append({
                    "step": "load_stats",
                    "success": False,
                    "error": str(e)
                })
            
            # Calculate workflow success
            successful_steps = sum(1 for step in workflow_steps if step.get("success", False))
            total_steps = len(workflow_steps)
            workflow_success_rate = successful_steps / total_steps if total_steps > 0 else 0
            
            result = {
                "test_name": "user_workflow_simulation",
                "success": workflow_success_rate >= 0.8,  # 80% of workflow must work
                "test_duration_ms": (time.time() - start_time) * 1000,
                "total_steps": total_steps,
                "successful_steps": successful_steps,
                "workflow_success_rate": workflow_success_rate,
                "workflow_details": workflow_steps,
                "error": None if workflow_success_rate >= 0.8 else f"Only {workflow_success_rate:.1%} of workflow working"
            }
            
        except Exception as e:
            result = {
                "test_name": "user_workflow_simulation",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000
            }
        
        self.test_results.append(result)
        return result
    
    def test_system_under_load(self) -> Dict[str, Any]:
        """Test system performance under concurrent load."""
        start_time = time.time()
        
        try:
            print("⚡ Testing system under concurrent load...")
            
            def simulate_user_session():
                """Simulate a single user session."""
                session_start = time.time()
                requests_made = 0
                errors = 0
                
                # Typical user session: check portfolio, prices, make some chart requests
                endpoints = [
                    "/api/portfolio",
                    "/api/prices", 
                    "/api/trades?limit=5",
                    "/api/chart/BTCUSDT",
                    "/api/stats"
                ]
                
                for endpoint in endpoints:
                    try:
                        response = requests.get(f"{self.backend_url}{endpoint}", timeout=10)
                        requests_made += 1
                        if response.status_code != 200:
                            errors += 1
                        # Small delay between requests
                        time.sleep(0.1)
                    except:
                        errors += 1
                
                return {
                    "session_duration": time.time() - session_start,
                    "requests_made": requests_made,
                    "errors": errors,
                    "success_rate": (requests_made - errors) / requests_made if requests_made > 0 else 0
                }
            
            # Run concurrent user sessions
            num_concurrent_users = 10
            with ThreadPoolExecutor(max_workers=num_concurrent_users) as executor:
                futures = [executor.submit(simulate_user_session) for _ in range(num_concurrent_users)]
                session_results = [future.result() for future in futures]
            
            # Analyze load test results
            total_requests = sum(result["requests_made"] for result in session_results)
            total_errors = sum(result["errors"] for result in session_results)
            avg_session_duration = sum(result["session_duration"] for result in session_results) / len(session_results)
            overall_success_rate = (total_requests - total_errors) / total_requests if total_requests > 0 else 0
            
            result = {
                "test_name": "system_under_load",
                "success": overall_success_rate >= 0.9 and avg_session_duration < 30,  # 90% success rate, under 30s per session
                "test_duration_ms": (time.time() - start_time) * 1000,
                "concurrent_users": num_concurrent_users,
                "total_requests": total_requests,
                "total_errors": total_errors,
                "overall_success_rate": overall_success_rate,
                "avg_session_duration": avg_session_duration,
                "session_results": session_results,
                "error": None if overall_success_rate >= 0.9 else f"Only {overall_success_rate:.1%} success rate under load"
            }
            
        except Exception as e:
            result = {
                "test_name": "system_under_load",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000
            }
        
        self.test_results.append(result)
        return result
    
    def test_error_recovery(self) -> Dict[str, Any]:
        """Test system error recovery and resilience."""
        start_time = time.time()
        
        try:
            print("🔧 Testing error recovery and resilience...")
            
            recovery_tests = []
            
            # Test 1: Invalid API requests
            invalid_requests = [
                "/api/nonexistent",
                "/api/chart/INVALID_SYMBOL",
                "/api/trades?limit=invalid",
                "/api/portfolio?user=nonexistent"
            ]
            
            for invalid_url in invalid_requests:
                try:
                    response = requests.get(f"{self.backend_url}{invalid_url}", timeout=5)
                    # Good error handling returns 4xx status codes
                    handles_error_properly = 400 <= response.status_code < 500
                    recovery_tests.append({
                        "test": f"invalid_request_{invalid_url.split('/')[-1]}",
                        "success": handles_error_properly,
                        "status_code": response.status_code,
                        "details": "Proper error status code" if handles_error_properly else "Unexpected status code"
                    })
                except Exception as e:
                    recovery_tests.append({
                        "test": f"invalid_request_{invalid_url.split('/')[-1]}",
                        "success": False,
                        "error": str(e)
                    })
            
            # Test 2: Malformed JSON requests
            try:
                response = requests.post(f"{self.backend_url}/api/portfolio", 
                                       data="invalid json",
                                       headers={"Content-Type": "application/json"},
                                       timeout=5)
                handles_bad_json = 400 <= response.status_code < 500
                recovery_tests.append({
                    "test": "malformed_json_request",
                    "success": handles_bad_json,
                    "status_code": response.status_code,
                    "details": "Handles bad JSON properly" if handles_bad_json else "Doesn't handle bad JSON"
                })
            except Exception as e:
                recovery_tests.append({
                    "test": "malformed_json_request",
                    "success": False,
                    "error": str(e)
                })
            
            # Test 3: System still responsive after errors
            try:
                response = requests.get(f"{self.backend_url}/health", timeout=5)
                system_responsive = response.status_code == 200
                recovery_tests.append({
                    "test": "system_responsive_after_errors",
                    "success": system_responsive,
                    "status_code": response.status_code,
                    "details": "System remains healthy after error tests"
                })
            except Exception as e:
                recovery_tests.append({
                    "test": "system_responsive_after_errors",
                    "success": False,
                    "error": str(e)
                })
            
            # Calculate recovery success rate
            successful_recovery_tests = sum(1 for test in recovery_tests if test.get("success", False))
            total_recovery_tests = len(recovery_tests)
            recovery_success_rate = successful_recovery_tests / total_recovery_tests if total_recovery_tests > 0 else 0
            
            result = {
                "test_name": "error_recovery",
                "success": recovery_success_rate >= 0.8,  # 80% of error handling should work
                "test_duration_ms": (time.time() - start_time) * 1000,
                "recovery_tests_run": total_recovery_tests,
                "recovery_tests_passed": successful_recovery_tests,
                "recovery_success_rate": recovery_success_rate,
                "recovery_details": recovery_tests,
                "error": None if recovery_success_rate >= 0.8 else f"Only {recovery_success_rate:.1%} error recovery working"
            }
            
        except Exception as e:
            result = {
                "test_name": "error_recovery",
                "success": False,
                "error": str(e),
                "test_duration_ms": (time.time() - start_time) * 1000
            }
        
        self.test_results.append(result)
        return result
    
    def run_all_e2e_tests(self) -> Dict[str, Any]:
        """Run all end-to-end integration tests."""
        print("🔗 Running End-to-End Integration Tests...")
        print("=" * 60)
        
        # Start full stack
        if not self.start_full_stack():
            return {
                "error": "Failed to start full stack for E2E testing",
                "timestamp": datetime.now().isoformat()
            }
        
        test_summary = {
            "start_time": datetime.now().isoformat(),
            "tests_run": 0,
            "tests_passed": 0,
            "tests_failed": 0,
            "total_duration_ms": 0
        }
        
        start_time = time.time()
        
        try:
            # E2E test suite
            e2e_tests = [
                ("Full Stack Communication", self.test_full_stack_communication),
                ("Real-time Data Flow", self.test_real_time_data_flow),
                ("User Workflow Simulation", self.test_user_workflow_simulation),
                ("System Under Load", self.test_system_under_load),
                ("Error Recovery", self.test_error_recovery)
            ]
            
            for test_name, test_func in e2e_tests:
                print(f"\n📋 Running E2E Test: {test_name}")
                
                try:
                    result = test_func()
                    test_summary["tests_run"] += 1
                    
                    if result.get("success", False):
                        test_summary["tests_passed"] += 1
                        print(f"  ✅ {test_name}")
                        
                        # Print key metrics for successful tests
                        if "success_rate" in result:
                            print(f"     Success Rate: {result['success_rate']:.1%}")
                        if "test_duration_ms" in result:
                            print(f"     Duration: {result['test_duration_ms']:.0f}ms")
                    else:
                        test_summary["tests_failed"] += 1
                        print(f"  ❌ {test_name}: {result.get('error', 'Failed')}")
                        
                except Exception as e:
                    test_summary["tests_run"] += 1
                    test_summary["tests_failed"] += 1
                    print(f"  ❌ {test_name}: {str(e)}")
            
            test_summary["total_duration_ms"] = (time.time() - start_time) * 1000
            test_summary["success_rate"] = test_summary["tests_passed"] / test_summary["tests_run"] if test_summary["tests_run"] > 0 else 0
            test_summary["end_time"] = datetime.now().isoformat()
            test_summary["detailed_results"] = self.test_results
            
            # Print final summary
            print("\n" + "=" * 60)
            print("🔗 End-to-End Integration Test Summary:")
            print(f"  Tests Run: {test_summary['tests_run']}")
            print(f"  Tests Passed: {test_summary['tests_passed']}")
            print(f"  Tests Failed: {test_summary['tests_failed']}")
            print(f"  Success Rate: {test_summary['success_rate']:.1%}")
            print(f"  Total Duration: {test_summary['total_duration_ms']:.0f}ms")
            
            return test_summary
            
        finally:
            # Always stop the full stack
            self.stop_full_stack()


def main():
    """Main E2E test execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="End-to-End Integration Tests")
    parser.add_argument("--output", help="Output file for test results")
    
    args = parser.parse_args()
    
    # Run E2E tests
    tester = E2ESystemTester()
    results = tester.run_all_e2e_tests()
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 E2E results saved to: {args.output}")
    
    # Exit with appropriate code
    success_rate = results.get("success_rate", 0)
    if success_rate >= 0.75:  # 75% success rate required for E2E
        print("✅ End-to-End integration tests PASSED")
        return 0
    else:
        print("❌ End-to-End integration tests FAILED")
        return 1


if __name__ == "__main__":
    exit(main())