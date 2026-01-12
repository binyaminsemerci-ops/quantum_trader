#!/usr/bin/env python3
"""
Golden Contract Audit Script

Validates systemd services and ops environment comply with the Golden Contract.

Exit codes:
  0 = All checks pass (compliant)
  1 = Violations found (non-compliant)
  2 = Critical error (cannot audit)
"""

import os
import sys
import subprocess
from pathlib import Path
from typing import List, Tuple

# ============================================================================
# CONFIGURATION (GOLDEN CONTRACT)
# ============================================================================

REPO_ROOT = Path("/home/qt/quantum_trader")
ENV_DIR = Path("/etc/quantum")
VENV_ROOT = Path("/opt/quantum/venvs")
DATA_DIR = Path("/opt/quantum/data")
SYSTEMD_DIR = Path("/etc/systemd/system")

SERVICES = {
    'ai-engine': {
        'venv': 'ai-engine',
        'env_file': 'ai-engine.env',
        'systemd_unit': 'quantum-ai-engine.service'
    },
    'backend': {
        'venv': 'backend',
        'env_file': 'backend.env',
        'systemd_unit': 'quantum-backend.service'
    },
    'execution': {
        'venv': 'execution',
        'env_file': 'execution.env',
        'systemd_unit': 'quantum-execution.service'
    },
    'rl-agent': {
        'venv': 'rl-agent',
        'env_file': 'rl-agent.env',
        'systemd_unit': 'quantum-rl-sizing.service'
    }
}

REQUIRED_ENV_VARS = [
    'REPO_ROOT',
    'DATA_DIR',
    'SERVICE_NAME',
    'REDIS_HOST',
    'REDIS_PORT'
]

# ============================================================================
# COLORS
# ============================================================================

class Colors:
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    BOLD = '\033[1m'
    NC = '\033[0m'

# ============================================================================
# AUDIT FUNCTIONS
# ============================================================================

def check_redis() -> Tuple[bool, str]:
    """Check Redis is running"""
    try:
        result = subprocess.run(
            ['redis-cli', 'PING'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0 and 'PONG' in result.stdout:
            return True, "Redis responding on localhost:6379"
        return False, "Redis not responding"
    except Exception as e:
        return False, f"Redis check failed: {e}"


def check_database() -> Tuple[bool, str]:
    """Check database exists"""
    db_path = DATA_DIR / "quantum_trader.db"
    if db_path.exists():
        size = db_path.stat().st_size
        return True, f"Database exists ({size} bytes)"
    return False, f"Database not found: {db_path}"


def check_repo() -> Tuple[bool, str]:
    """Check repo exists"""
    if REPO_ROOT.exists() and REPO_ROOT.is_dir():
        git_dir = REPO_ROOT / ".git"
        if git_dir.exists():
            return True, f"Repo exists: {REPO_ROOT}"
        return False, f"Repo exists but not a git repo: {REPO_ROOT}"
    return False, f"Repo not found: {REPO_ROOT}"


def check_venv(service_name: str, venv_name: str) -> Tuple[bool, str]:
    """Check venv exists"""
    venv_path = VENV_ROOT / venv_name
    python_bin = venv_path / "bin" / "python"
    
    if not venv_path.exists():
        return False, f"Venv not found: {venv_path}"
    
    if not python_bin.exists():
        return False, f"Python not found: {python_bin}"
    
    # Get Python version
    try:
        result = subprocess.run(
            [str(python_bin), '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        version = result.stdout.strip()
        return True, f"Venv OK: {venv_path} ({version})"
    except Exception as e:
        return False, f"Venv check failed: {e}"


def check_env_file(env_filename: str) -> Tuple[bool, str]:
    """Check env file exists and has required vars"""
    env_path = ENV_DIR / env_filename
    
    if not env_path.exists():
        return False, f"Env file not found: {env_path}"
    
    # Check permissions (should be 640 or 644)
    stat = env_path.stat()
    mode = oct(stat.st_mode)[-3:]
    if mode not in ['640', '644', '600']:
        return False, f"Env file has wrong permissions: {mode} (should be 640)"
    
    # Read and check required vars
    try:
        with open(env_path) as f:
            content = f.read()
        
        missing_vars = []
        for var in REQUIRED_ENV_VARS:
            if f"{var}=" not in content:
                missing_vars.append(var)
        
        if missing_vars:
            return False, f"Missing vars: {', '.join(missing_vars)}"
        
        return True, f"Env file OK: {env_path}"
    except Exception as e:
        return False, f"Env file read failed: {e}"


def check_systemd_unit(unit_name: str, service_name: str, venv_name: str, env_filename: str) -> Tuple[bool, str]:
    """Check systemd unit follows golden contract"""
    unit_path = SYSTEMD_DIR / unit_name
    
    if not unit_path.exists():
        return False, f"Unit not found: {unit_path}"
    
    try:
        with open(unit_path) as f:
            content = f.read()
        
        violations = []
        
        # Check WorkingDirectory
        if f"WorkingDirectory={REPO_ROOT}" not in content:
            violations.append(f"WorkingDirectory not {REPO_ROOT}")
        
        # Check EnvironmentFile
        env_path = ENV_DIR / env_filename
        if f"EnvironmentFile={env_path}" not in content:
            violations.append(f"EnvironmentFile not {env_path}")
        
        # Check User
        if "User=qt" not in content:
            violations.append("User not qt")
        
        # Check venv path in ExecStart
        venv_python = f"{VENV_ROOT}/{venv_name}/bin/python"
        if venv_python not in content:
            violations.append(f"ExecStart not using {venv_python}")
        
        # Check After Redis
        if "After=" not in content or "redis-server.service" not in content:
            violations.append("After= missing redis-server.service")
        
        # Check StandardOutput
        if "StandardOutput=journal" not in content:
            violations.append("StandardOutput not journal")
        
        if violations:
            return False, f"Unit violations: {', '.join(violations)}"
        
        return True, f"Unit OK: {unit_path}"
    except Exception as e:
        return False, f"Unit read failed: {e}"


def check_ops_wrapper() -> Tuple[bool, str]:
    """Check ops/run.sh wrapper exists and is executable"""
    wrapper_path = REPO_ROOT / "ops" / "run.sh"
    
    if not wrapper_path.exists():
        return False, f"Wrapper not found: {wrapper_path}"
    
    if not os.access(wrapper_path, os.X_OK):
        return False, f"Wrapper not executable: {wrapper_path}"
    
    return True, f"Wrapper OK: {wrapper_path}"


def check_makefile() -> Tuple[bool, str]:
    """Check Makefile uses ops/run.sh"""
    makefile_path = REPO_ROOT / "Makefile"
    
    if not makefile_path.exists():
        return False, f"Makefile not found: {makefile_path}"
    
    try:
        with open(makefile_path) as f:
            content = f.read()
        
        violations = []
        
        # Check RUN variable
        if "RUN :=" not in content and "RUN =" not in content:
            violations.append("RUN variable not defined")
        
        # Check ops/run.sh usage
        if "ops/run.sh" not in content:
            violations.append("ops/run.sh not used")
        
        # Check no direct Python calls
        if "/bin/python " in content and "$(RUN)" not in content:
            violations.append("Direct Python calls found (should use $(RUN))")
        
        if violations:
            return False, f"Makefile violations: {', '.join(violations)}"
        
        return True, f"Makefile OK: {makefile_path}"
    except Exception as e:
        return False, f"Makefile read failed: {e}"


# ============================================================================
# MAIN AUDIT
# ============================================================================

def main():
    print(f"{Colors.BOLD}{'='*70}{Colors.NC}")
    print(f"{Colors.BOLD}GOLDEN CONTRACT AUDIT{Colors.NC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.NC}")
    print()
    
    all_pass = True
    
    # ========================================================================
    # INFRASTRUCTURE CHECKS
    # ========================================================================
    
    print(f"{Colors.BLUE}📦 Infrastructure Checks{Colors.NC}")
    print()
    
    checks = [
        ("Repo", check_repo()),
        ("Redis", check_redis()),
        ("Database", check_database()),
        ("Ops Wrapper", check_ops_wrapper()),
        ("Makefile", check_makefile())
    ]
    
    for name, (passed, message) in checks:
        icon = f"{Colors.GREEN}✅{Colors.NC}" if passed else f"{Colors.RED}❌{Colors.NC}"
        print(f"{icon} {name}: {message}")
        if not passed:
            all_pass = False
    
    print()
    
    # ========================================================================
    # SERVICE CHECKS
    # ========================================================================
    
    print(f"{Colors.BLUE}🔧 Service Checks{Colors.NC}")
    print()
    
    for service_name, config in SERVICES.items():
        print(f"{Colors.BOLD}{service_name}:{Colors.NC}")
        
        # Venv check
        passed, message = check_venv(service_name, config['venv'])
        icon = f"{Colors.GREEN}✅{Colors.NC}" if passed else f"{Colors.RED}❌{Colors.NC}"
        print(f"  {icon} Venv: {message}")
        if not passed:
            all_pass = False
        
        # Env file check
        passed, message = check_env_file(config['env_file'])
        icon = f"{Colors.GREEN}✅{Colors.NC}" if passed else f"{Colors.RED}❌{Colors.NC}"
        print(f"  {icon} Env: {message}")
        if not passed:
            all_pass = False
        
        # Systemd unit check
        passed, message = check_systemd_unit(
            config['systemd_unit'],
            service_name,
            config['venv'],
            config['env_file']
        )
        icon = f"{Colors.GREEN}✅{Colors.NC}" if passed else f"{Colors.RED}❌{Colors.NC}"
        print(f"  {icon} Unit: {message}")
        if not passed:
            all_pass = False
        
        print()
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    
    print(f"{Colors.BOLD}{'='*70}{Colors.NC}")
    
    if all_pass:
        print(f"{Colors.GREEN}{Colors.BOLD}✅ ALL CHECKS PASSED - COMPLIANT{Colors.NC}")
        print()
        print("The system follows the Golden Contract.")
        print("All services can be deployed safely.")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}❌ VIOLATIONS FOUND - NON-COMPLIANT{Colors.NC}")
        print()
        print("The system DOES NOT follow the Golden Contract.")
        print("Fix violations before deploying.")
        return 1


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Audit interrupted{Colors.NC}")
        sys.exit(2)
    except Exception as e:
        print(f"\n{Colors.RED}❌ CRITICAL ERROR: {e}{Colors.NC}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
