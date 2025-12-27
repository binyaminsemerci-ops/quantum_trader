"""
SYSTEM HEALTH REPORT GENERATOR
Generates a comprehensive health report of the Quantum Trader system
"""
import subprocess
import json
import re
from datetime import datetime
from typing import Dict, List

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
    print(f"{Colors.BOLD}{Colors.BLUE}📊 {text}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'─'*80}{Colors.RESET}\n")

def run_docker_command(cmd: str) -> str:
    """Run docker command and return output"""
    try:
        result = subprocess.run(
            f"docker {cmd}",
            shell=True,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace'
        )
        return result.stdout if result.stdout else ""
    except Exception as e:
        return f"Error: {e}"

def get_container_stats() -> Dict:
    """Get container resource usage"""
    print_section("DOCKER CONTAINER STATUS")
    
    # Container status
    status_output = run_docker_command(
        'ps --format "{{.Names}}|{{.Status}}|{{.Ports}}"'
    )
    
    containers = []
    for line in status_output.strip().split('\n'):
        if '|' in line:
            name, status, ports = line.split('|')
            containers.append({
                "name": name.strip(),
                "status": status.strip(),
                "ports": ports.strip()
            })
            
            # Determine status color
            if "Up" in status:
                status_color = Colors.GREEN
                status_icon = "✅"
            elif "Restarting" in status:
                status_color = Colors.YELLOW
                status_icon = "⚠️"
            else:
                status_color = Colors.RED
                status_icon = "❌"
            
            print(f"{status_icon} {Colors.BOLD}{name.strip()}{Colors.RESET}")
            print(f"   Status: {status_color}{status.strip()}{Colors.RESET}")
            if ports.strip():
                print(f"   Ports: {ports.strip()}")
    
    # Resource usage
    stats_output = run_docker_command(
        'stats --no-stream --format "{{.Name}}|{{.CPUPerc}}|{{.MemUsage}}|{{.MemPerc}}"'
    )
    
    print(f"\n{Colors.BOLD}Resource Usage:{Colors.RESET}\n")
    for line in stats_output.strip().split('\n'):
        if '|' in line:
            name, cpu, mem_usage, mem_perc = line.split('|')
            print(f"  {Colors.CYAN}•{Colors.RESET} {name.strip()}")
            print(f"    CPU: {cpu.strip()}")
            print(f"    Memory: {mem_usage.strip()} ({mem_perc.strip()})")
    
    return containers

def analyze_logs(time_window: str = "5m") -> Dict:
    """Analyze backend logs"""
    print_section(f"LOG ANALYSIS (Last {time_window})")
    
    # Get logs
    logs = run_docker_command(f'logs quantum_backend --since {time_window} 2>&1')
    
    # Count different log levels
    errors = len(re.findall(r'"level":\s*"ERROR"', logs))
    warnings = len(re.findall(r'"level":\s*"WARNING"', logs))
    info = len(re.findall(r'"level":\s*"INFO"', logs))
    
    # Count trades
    trades = len(re.findall(r'Order placed:', logs))
    approved_trades = len(re.findall(r'TRADE APPROVED:', logs))
    
    # Count positions
    position_checks = re.findall(r'Position check:\s*(\d+)\s*total', logs)
    current_positions = int(position_checks[-1]) if position_checks else 0
    
    # Health checks
    health_critical = len(re.findall(r'Status: CRITICAL', logs))
    health_degraded = len(re.findall(r'Status: DEGRADED', logs))
    health_ok = len(re.findall(r'Status: OK', logs))
    
    print(f"{Colors.BOLD}Log Statistics:{Colors.RESET}\n")
    
    # Log levels
    print(f"  {Colors.RED if errors > 50 else Colors.YELLOW if errors > 0 else Colors.GREEN}•{Colors.RESET} Errors: {errors}")
    print(f"  {Colors.YELLOW if warnings > 100 else Colors.GREEN}•{Colors.RESET} Warnings: {warnings}")
    print(f"  {Colors.GREEN}•{Colors.RESET} Info: {info}")
    
    print(f"\n{Colors.BOLD}Trading Activity:{Colors.RESET}\n")
    print(f"  {Colors.CYAN}•{Colors.RESET} Trade approvals: {approved_trades}")
    print(f"  {Colors.CYAN}•{Colors.RESET} Orders placed: {trades}")
    print(f"  {Colors.CYAN}•{Colors.RESET} Current positions: {current_positions}")
    
    print(f"\n{Colors.BOLD}Health Status:{Colors.RESET}\n")
    if health_critical > 0:
        print(f"  {Colors.RED}❌ Critical status detected: {health_critical} times{Colors.RESET}")
    if health_degraded > 0:
        print(f"  {Colors.YELLOW}⚠️  Degraded status detected: {health_degraded} times{Colors.RESET}")
    if health_ok > 0:
        print(f"  {Colors.GREEN}✅ Healthy status: {health_ok} times{Colors.RESET}")
    
    # Check for common issues
    print(f"\n{Colors.BOLD}Common Issues:{Colors.RESET}\n")
    
    connection_pool_warnings = len(re.findall(r'Connection pool is full', logs))
    if connection_pool_warnings > 0:
        print(f"  {Colors.YELLOW}⚠️  Connection pool full warnings: {connection_pool_warnings}{Colors.RESET}")
    
    universe_critical = len(re.findall(r'universe_os is critical', logs))
    if universe_critical > 0:
        print(f"  {Colors.RED}❌ Universe OS critical: {universe_critical} times{Colors.RESET}")
    
    api_errors = len(re.findall(r'APIError', logs))
    if api_errors > 0:
        print(f"  {Colors.RED}❌ API Errors: {api_errors}{Colors.RESET}")
    
    if connection_pool_warnings == 0 and universe_critical == 0 and api_errors == 0:
        print(f"  {Colors.GREEN}✅ No critical issues detected{Colors.RESET}")
    
    return {
        "errors": errors,
        "warnings": warnings,
        "trades": trades,
        "positions": current_positions,
        "health_critical": health_critical,
        "health_degraded": health_degraded,
        "health_ok": health_ok
    }

def calculate_health_score(log_stats: Dict, containers: List[Dict]) -> int:
    """Calculate overall health score (0-100)"""
    score = 100
    
    # Deduct for errors
    score -= min(log_stats["errors"] * 0.1, 20)
    
    # Deduct for critical health
    score -= log_stats["health_critical"] * 10
    
    # Deduct for degraded health
    score -= log_stats["health_degraded"] * 5
    
    # Deduct for stopped containers
    stopped = sum(1 for c in containers if "Up" not in c["status"])
    score -= stopped * 20
    
    # Bonus for active trading
    if log_stats["trades"] > 5:
        score += 5
    
    return max(0, min(100, int(score)))

def generate_health_report():
    """Generate comprehensive health report"""
    print_header("QUANTUM TRADER - SYSTEM HEALTH REPORT")
    
    print(f"{Colors.BOLD}Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}{Colors.RESET}\n")
    
    # Collect data
    containers = get_container_stats()
    log_stats = analyze_logs("5m")
    
    # Calculate health score
    health_score = calculate_health_score(log_stats, containers)
    
    # Overall health assessment
    print_section("OVERALL HEALTH ASSESSMENT")
    
    if health_score >= 90:
        health_color = Colors.GREEN
        health_status = "EXCELLENT"
        health_icon = "🎉"
    elif health_score >= 75:
        health_color = Colors.GREEN
        health_status = "GOOD"
        health_icon = "✅"
    elif health_score >= 60:
        health_color = Colors.YELLOW
        health_status = "FAIR"
        health_icon = "⚠️"
    elif health_score >= 40:
        health_color = Colors.YELLOW
        health_status = "DEGRADED"
        health_icon = "⚠️"
    else:
        health_color = Colors.RED
        health_status = "CRITICAL"
        health_icon = "❌"
    
    print(f"{health_icon} {Colors.BOLD}System Health Score: {health_color}{health_score}/100{Colors.RESET}")
    print(f"   Status: {health_color}{Colors.BOLD}{health_status}{Colors.RESET}\n")
    
    # Recommendations
    print_section("RECOMMENDATIONS")
    
    recommendations = []
    
    if log_stats["errors"] > 50:
        recommendations.append(f"{Colors.YELLOW}• Investigate high error count (restart may be needed){Colors.RESET}")
    
    if log_stats["health_critical"] > 0:
        recommendations.append(f"{Colors.RED}• Address critical health issues immediately{Colors.RESET}")
    
    if log_stats["trades"] == 0:
        recommendations.append(f"{Colors.YELLOW}• No trades in last 5 minutes - check trading conditions{Colors.RESET}")
    
    any_stopped = any("Up" not in c["status"] for c in containers)
    if any_stopped:
        recommendations.append(f"{Colors.RED}• Restart stopped containers{Colors.RESET}")
    
    if not recommendations:
        print(f"{Colors.GREEN}✅ No immediate action required - system operating normally{Colors.RESET}\n")
    else:
        for rec in recommendations:
            print(rec)
        print()
    
    print("="*80 + "\n")

if __name__ == "__main__":
    try:
        generate_health_report()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Report generation interrupted{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.RED}Error generating report: {e}{Colors.RESET}\n")
