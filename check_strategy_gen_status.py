"""
Quick status checker for Strategy Generator AI services
"""
import subprocess
import sys
from datetime import datetime

def run_command(cmd):
    """Run a command and return output"""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            capture_output=True, 
            text=True,
            timeout=10
        )
        return result.stdout + result.stderr
    except Exception as e:
        return f"Error: {str(e)}"

def main():
    print("=" * 60)
    print(f"Strategy Generator AI - Status Check")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Check Docker containers
    print("\n🐳 Docker Containers:")
    print("-" * 60)
    containers = run_command(
        'docker ps --filter "name=quantum_" --format "{{.Names}}|{{.Status}}|{{.Ports}}"'
    )
    if containers.strip():
        for line in containers.strip().split('\n'):
            if '|' in line:
                name, status, ports = line.split('|', 2)
                status_emoji = "✅" if "Up" in status else "❌" if "Exited" in status else "🔄"
                print(f"{status_emoji} {name}")
                print(f"   Status: {status}")
                if ports:
                    print(f"   Ports: {ports}")
    else:
        print("No services running")
    
    # Check metrics endpoint
    print("\n📊 Metrics Endpoint:")
    print("-" * 60)
    metrics = run_command("curl -s http://localhost:9090/metrics 2>&1")
    if "strategy_generator" in metrics:
        print("✅ Metrics server responding")
        # Count some key metrics
        for metric in ["generations_total", "strategies_generated", "active_live_strategies"]:
            if metric in metrics:
                print(f"   Found metric: {metric}")
    else:
        print("❌ Metrics server not responding")
    
    # Check database status
    print("\n💾 Database Status:")
    print("-" * 60)
    try:
        from backend.research.postgres_repository import PostgresStrategyRepository
        repo = PostgresStrategyRepository()
        
        candidate = len(repo.get_by_status("CANDIDATE"))
        shadow = len(repo.get_by_status("SHADOW"))
        live = len(repo.get_by_status("LIVE"))
        disabled = len(repo.get_by_status("DISABLED"))
        
        print(f"✅ Database connected")
        print(f"   CANDIDATE: {candidate}")
        print(f"   SHADOW: {shadow}")
        print(f"   LIVE: {live}")
        print(f"   DISABLED: {disabled}")
        print(f"   Total: {candidate + shadow + live + disabled}")
    except Exception as e:
        print(f"❌ Database error: {str(e)}")
    
    # Check recent logs
    print("\n📋 Recent Logs (last 5 lines each):")
    print("-" * 60)
    for container in ["quantum-strategy-generator", "quantum-shadow-tester", "quantum-metrics"]:
        print(f"\n{container}:")
        logs = run_command(f"journalctl -u {container}.service -n 5 --no-pager 2>&1")
        if logs:
            for line in logs.split('\n')[:5]:
                if line.strip():
                    print(f"   {line[:100]}")
    
    print("\n" + "=" * 60)
    print("Status check complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
