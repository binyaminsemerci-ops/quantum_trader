import subprocess
import sys

print("=== Deploying EXIT_GUARD via Python SSH ===")

# Read the deployment script
with open(r"C:\quantum_trader\deploy_exit_guard_final.sh", "r") as f:
    script_content = f.read()

# Upload via SSH stdin
cmd = [
    "wsl", "bash", "-c",
    "ssh -i ~/.ssh/hetzner_fresh root@46.224.116.254 'cat > /tmp/deploy_guard.sh && chmod +x /tmp/deploy_guard.sh && bash /tmp/deploy_guard.sh'"
]

try:
    result = subprocess.run(
        cmd,
        input=script_content.encode(),
        capture_output=True,
        timeout=60
    )
    
    print("STDOUT:")
    print(result.stdout.decode())
    
    if result.stderr:
        print("\nSTDERR:")
        print(result.stderr.decode())
    
    print(f"\nExit code: {result.returncode}")
    
except subprocess.TimeoutExpired:
    print("❌ Command timed out after 60 seconds")
    sys.exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)
