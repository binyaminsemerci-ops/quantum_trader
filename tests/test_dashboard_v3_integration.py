"""
Dashboard V3.0 Integration Test
DASHBOARD-V3-001: Phase 11 - End-to-End Integration Testing

Tests:
- Backend API endpoints are accessible
- Frontend components can fetch data
- WebSocket connection works
- Real-time updates flow correctly
"""

import asyncio
import json
import requests
import time
from typing import Dict, Any
import websockets

# Configuration
BACKEND_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws/dashboard"
TIMEOUT = 10  # seconds


class Colors:
    """Terminal colors"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'


class DashboardIntegrationTest:
    """Integration test suite for Dashboard V3.0"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
    
    def run_all_tests(self):
        """Run all integration tests"""
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BLUE}Dashboard V3.0 Integration Test Suite{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")
        
        # Backend API Tests
        print(f"{Colors.CYAN}[Phase 1/4] Backend API Connectivity{Colors.RESET}")
        self.test_backend_health()
        self.test_overview_endpoint()
        self.test_trading_endpoint()
        self.test_risk_endpoint()
        self.test_system_endpoint()
        print()
        
        # Data Consistency Tests
        print(f"{Colors.CYAN}[Phase 2/4] Data Consistency{Colors.RESET}")
        self.test_timestamp_consistency()
        self.test_data_types()
        print()
        
        # Performance Tests
        print(f"{Colors.CYAN}[Phase 3/4] Performance{Colors.RESET}")
        self.test_response_times()
        print()
        
        # WebSocket Tests
        print(f"{Colors.CYAN}[Phase 4/4] WebSocket Real-Time Updates{Colors.RESET}")
        self.test_websocket_connection()
        print()
        
        self.print_summary()
    
    def test_backend_health(self):
        """Test backend health endpoint"""
        try:
            response = requests.get(f"{BACKEND_URL}/health", timeout=TIMEOUT)
            if response.status_code == 200:
                self.pass_test("Backend health check")
            else:
                self.fail_test(f"Backend health check (status {response.status_code})")
        except requests.exceptions.RequestException as e:
            self.fail_test(f"Backend health check - {str(e)}")
    
    def test_overview_endpoint(self):
        """Test /api/dashboard/overview endpoint"""
        try:
            response = requests.get(f"{BACKEND_URL}/api/dashboard/overview", timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                # Verify required fields
                required_fields = ["timestamp", "go_live_active", "global_pnl", "risk_state"]
                missing = [f for f in required_fields if f not in data]
                
                if missing:
                    self.fail_test(f"Overview endpoint - missing fields: {missing}")
                else:
                    self.pass_test("Overview endpoint (all fields present)")
                    
                    # Show sample data
                    print(f"    GO-LIVE: {data.get('go_live_active', 'N/A')}")
                    print(f"    Risk State: {data.get('risk_state', 'N/A')}")
            else:
                self.fail_test(f"Overview endpoint (status {response.status_code})")
        
        except requests.exceptions.RequestException as e:
            self.fail_test(f"Overview endpoint - {str(e)}")
    
    def test_trading_endpoint(self):
        """Test /api/dashboard/trading endpoint"""
        try:
            response = requests.get(f"{BACKEND_URL}/api/dashboard/trading", timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                required_fields = ["open_positions", "recent_orders", "recent_signals"]
                missing = [f for f in required_fields if f not in data]
                
                if missing:
                    self.fail_test(f"Trading endpoint - missing fields: {missing}")
                else:
                    self.pass_test("Trading endpoint (all fields present)")
                    
                    # Show counts
                    pos_count = len(data.get("open_positions", []))
                    orders_count = len(data.get("recent_orders", []))
                    signals_count = len(data.get("recent_signals", []))
                    print(f"    Positions: {pos_count}, Orders: {orders_count}, Signals: {signals_count}")
            else:
                self.fail_test(f"Trading endpoint (status {response.status_code})")
        
        except requests.exceptions.RequestException as e:
            self.fail_test(f"Trading endpoint - {str(e)}")
    
    def test_risk_endpoint(self):
        """Test /api/dashboard/risk endpoint"""
        try:
            response = requests.get(f"{BACKEND_URL}/api/dashboard/risk", timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                required_fields = ["risk_gate_decisions_stats", "ess_triggers_recent", "var_es_snapshot"]
                missing = [f for f in required_fields if f not in data]
                
                if missing:
                    self.fail_test(f"Risk endpoint - missing fields: {missing}")
                else:
                    self.pass_test("Risk endpoint (all fields present)")
                    
                    # Show ESS status
                    ess_count = len(data.get("ess_triggers_recent", []))
                    print(f"    ESS Triggers (24h): {ess_count}")
            else:
                self.fail_test(f"Risk endpoint (status {response.status_code})")
        
        except requests.exceptions.RequestException as e:
            self.fail_test(f"Risk endpoint - {str(e)}")
    
    def test_system_endpoint(self):
        """Test /api/dashboard/system endpoint"""
        try:
            response = requests.get(f"{BACKEND_URL}/api/dashboard/system", timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                required_fields = ["services_health", "exchanges_health", "failover_events_recent"]
                missing = [f for f in required_fields if f not in data]
                
                if missing:
                    self.fail_test(f"System endpoint - missing fields: {missing}")
                else:
                    self.pass_test("System endpoint (all fields present)")
                    
                    # Show service count
                    services = len(data.get("services_health", []))
                    exchanges = len(data.get("exchanges_health", []))
                    print(f"    Services: {services}, Exchanges: {exchanges}")
            else:
                self.fail_test(f"System endpoint (status {response.status_code})")
        
        except requests.exceptions.RequestException as e:
            self.fail_test(f"System endpoint - {str(e)}")
    
    def test_timestamp_consistency(self):
        """Test that timestamps are consistent across endpoints"""
        try:
            endpoints = ["overview", "trading", "risk", "system"]
            timestamps = []
            
            for endpoint in endpoints:
                response = requests.get(f"{BACKEND_URL}/api/dashboard/{endpoint}", timeout=TIMEOUT)
                if response.status_code == 200:
                    data = response.json()
                    if "timestamp" in data:
                        timestamps.append(data["timestamp"])
            
            if len(timestamps) >= 2:
                # All timestamps should be within 60 seconds of each other
                from datetime import datetime
                times = [datetime.fromisoformat(ts.replace('Z', '+00:00')) for ts in timestamps]
                max_diff = max(times) - min(times)
                
                if max_diff.total_seconds() < 60:
                    self.pass_test(f"Timestamp consistency (max diff: {max_diff.total_seconds():.1f}s)")
                else:
                    self.warn_test(f"Timestamp drift detected: {max_diff.total_seconds():.1f}s")
            else:
                self.warn_test("Not enough timestamps to compare")
        
        except Exception as e:
            self.fail_test(f"Timestamp consistency check - {str(e)}")
    
    def test_data_types(self):
        """Test that data types are correct"""
        try:
            response = requests.get(f"{BACKEND_URL}/api/dashboard/overview", timeout=TIMEOUT)
            
            if response.status_code == 200:
                data = response.json()
                
                # Type checks
                checks = [
                    ("go_live_active", bool),
                    ("global_pnl", dict),
                    ("risk_state", str),
                ]
                
                all_correct = True
                for field, expected_type in checks:
                    if field in data:
                        if isinstance(data[field], expected_type):
                            continue
                        else:
                            all_correct = False
                            print(f"    Type error: {field} is {type(data[field])}, expected {expected_type}")
                
                if all_correct:
                    self.pass_test("Data type validation")
                else:
                    self.fail_test("Data type validation")
        
        except Exception as e:
            self.fail_test(f"Data type validation - {str(e)}")
    
    def test_response_times(self):
        """Test API response times"""
        endpoints = {
            "overview": "/api/dashboard/overview",
            "trading": "/api/dashboard/trading",
            "risk": "/api/dashboard/risk",
            "system": "/api/dashboard/system",
        }
        
        times = {}
        for name, path in endpoints.items():
            try:
                start = time.time()
                response = requests.get(f"{BACKEND_URL}{path}", timeout=TIMEOUT)
                elapsed = time.time() - start
                times[name] = elapsed
                
                if elapsed < 5.0:
                    status = f"{Colors.GREEN}FAST{Colors.RESET}"
                elif elapsed < 10.0:
                    status = f"{Colors.YELLOW}OK{Colors.RESET}"
                else:
                    status = f"{Colors.RED}SLOW{Colors.RESET}"
                
                print(f"    {name:15} {elapsed:6.3f}s  {status}")
            
            except Exception as e:
                print(f"    {name:15} FAILED - {str(e)}")
        
        avg_time = sum(times.values()) / len(times) if times else 0
        if avg_time < 5.0:
            self.pass_test(f"Response times (avg: {avg_time:.3f}s)")
        else:
            self.warn_test(f"Response times slow (avg: {avg_time:.3f}s)")
    
    def test_websocket_connection(self):
        """Test WebSocket connection"""
        async def connect():
            try:
                async with websockets.connect(WS_URL, timeout=TIMEOUT) as ws:
                    self.pass_test("WebSocket connection")
                    
                    # Wait for initial message
                    try:
                        message = await asyncio.wait_for(ws.recv(), timeout=5.0)
                        data = json.loads(message)
                        self.pass_test(f"WebSocket message received (type: {data.get('type', 'unknown')})")
                    except asyncio.TimeoutError:
                        self.warn_test("WebSocket message timeout (no messages within 5s)")
            
            except Exception as e:
                self.fail_test(f"WebSocket connection - {str(e)}")
        
        try:
            asyncio.run(connect())
        except Exception as e:
            self.fail_test(f"WebSocket test - {str(e)}")
    
    def pass_test(self, name: str):
        """Record passed test"""
        self.passed += 1
        print(f"  {Colors.GREEN}✓{Colors.RESET} {name}")
    
    def fail_test(self, name: str):
        """Record failed test"""
        self.failed += 1
        print(f"  {Colors.RED}✗{Colors.RESET} {name}")
    
    def warn_test(self, name: str):
        """Record warning"""
        self.warnings += 1
        print(f"  {Colors.YELLOW}⚠{Colors.RESET} {name}")
    
    def print_summary(self):
        """Print test summary"""
        total = self.passed + self.failed + self.warnings
        
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}")
        print(f"{Colors.BLUE}Test Summary{Colors.RESET}")
        print(f"{Colors.BLUE}{'='*70}{Colors.RESET}\n")
        
        print(f"{Colors.GREEN}✓ Passed:   {self.passed}/{total}{Colors.RESET}")
        print(f"{Colors.YELLOW}⚠ Warnings: {self.warnings}/{total}{Colors.RESET}")
        print(f"{Colors.RED}✗ Failed:   {self.failed}/{total}{Colors.RESET}\n")
        
        if self.failed == 0:
            print(f"{Colors.GREEN}{'='*70}{Colors.RESET}")
            print(f"{Colors.GREEN}✓ ALL INTEGRATION TESTS PASSED{Colors.RESET}")
            print(f"{Colors.GREEN}{'='*70}{Colors.RESET}\n")
        else:
            print(f"{Colors.RED}{'='*70}{Colors.RESET}")
            print(f"{Colors.RED}✗ SOME TESTS FAILED - Check backend is running{Colors.RESET}")
            print(f"{Colors.RED}{'='*70}{Colors.RESET}\n")


def main():
    """Run integration tests"""
    print("\nStarting Dashboard V3.0 Integration Tests...")
    print(f"Backend URL: {BACKEND_URL}")
    print(f"WebSocket URL: {WS_URL}\n")
    
    tester = DashboardIntegrationTest()
    tester.run_all_tests()


if __name__ == "__main__":
    main()
