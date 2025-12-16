"""
Production Readiness Setup Script
==================================
Implements the 3 critical improvements from QA Report:
1. HTTPS/SSL Configuration
2. JWT Authentication System
3. P99 Latency Optimization (Caching)

Usage:
    python scripts/setup_production.py [--generate-cert] [--install-deps] [--configure-env]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def print_banner():
    """Print setup banner."""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║        QUANTUM TRADER v2.0 - PRODUCTION SETUP               ║
║        Implementing Critical QA Improvements                 ║
╚══════════════════════════════════════════════════════════════╝

This script will:
✅ Install security dependencies (JWT, Redis, SSL)
✅ Generate SSL certificates for HTTPS
✅ Configure environment variables
✅ Validate installation

"""
    print(banner)


def install_dependencies():
    """Install required dependencies."""
    print("\n📦 Installing security dependencies...")
    print("=" * 60)
    
    requirements_file = Path("requirements_security.txt")
    if not requirements_file.exists():
        print("❌ requirements_security.txt not found!")
        return False
    
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
            check=True
        )
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False


def generate_ssl_certificate():
    """Generate self-signed SSL certificate for development."""
    print("\n🔐 Generating SSL certificate...")
    print("=" * 60)
    
    cert_dir = Path("certs")
    cert_dir.mkdir(exist_ok=True)
    
    key_file = cert_dir / "key.pem"
    cert_file = cert_dir / "cert.pem"
    
    if key_file.exists() and cert_file.exists():
        print("ℹ️  Certificates already exist. Skipping generation.")
        overwrite = input("   Overwrite existing certificates? (y/N): ").strip().lower()
        if overwrite != 'y':
            return True
    
    try:
        # Generate self-signed certificate
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:4096",
            "-keyout", str(key_file),
            "-out", str(cert_file),
            "-days", "365",
            "-nodes",
            "-subj", "/CN=localhost/O=Quantum Trader/C=US"
        ], check=True, capture_output=True)
        
        print(f"✅ Certificate created: {cert_file}")
        print(f"✅ Private key created: {key_file}")
        print("⚠️  Note: Self-signed certificate for development only!")
        print("   For production, use Let's Encrypt or a proper CA.")
        return True
    except FileNotFoundError:
        print("❌ OpenSSL not found!")
        print("   Install OpenSSL:")
        print("   - Windows: https://slproweb.com/products/Win32OpenSSL.html")
        print("   - macOS: brew install openssl")
        print("   - Linux: sudo apt install openssl")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate certificate: {e}")
        print(f"   Output: {e.stderr.decode() if e.stderr else 'N/A'}")
        return False


def configure_environment():
    """Configure .env file with security settings."""
    print("\n⚙️  Configuring environment...")
    print("=" * 60)
    
    env_file = Path(".env")
    env_security = Path(".env.security")
    
    if not env_security.exists():
        print("❌ .env.security template not found!")
        return False
    
    # Read existing .env
    existing_env = {}
    if env_file.exists():
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    existing_env[key.strip()] = value.strip()
    
    # Generate secure keys if not present
    import secrets
    
    if 'JWT_SECRET_KEY' not in existing_env or existing_env['JWT_SECRET_KEY'].startswith('CHANGE_THIS'):
        jwt_secret = secrets.token_urlsafe(32)
        existing_env['JWT_SECRET_KEY'] = jwt_secret
        print(f"✅ Generated JWT_SECRET_KEY: {jwt_secret[:10]}...")
    
    if 'API_KEY_ADMIN' not in existing_env or existing_env['API_KEY_ADMIN'].startswith('CHANGE_THIS'):
        api_key_admin = secrets.token_urlsafe(32)
        existing_env['API_KEY_ADMIN'] = api_key_admin
        print(f"✅ Generated API_KEY_ADMIN: {api_key_admin[:10]}...")
    
    if 'API_KEY_USER' not in existing_env or existing_env['API_KEY_USER'].startswith('CHANGE_THIS'):
        api_key_user = secrets.token_urlsafe(32)
        existing_env['API_KEY_USER'] = api_key_user
        print(f"✅ Generated API_KEY_USER: {api_key_user[:10]}...")
    
    # Set defaults
    defaults = {
        'REDIS_URL': 'redis://localhost:6379',
        'JWT_ALGORITHM': 'HS256',
        'JWT_ACCESS_TOKEN_EXPIRE_MINUTES': '30',
        'JWT_REFRESH_TOKEN_EXPIRE_DAYS': '7',
        'RATE_LIMIT_REQUESTS': '100',
        'RATE_LIMIT_WINDOW_SECONDS': '60',
        'FORCE_HTTPS': 'false',  # Start with false for testing
        'SSL_CERTFILE': 'certs/cert.pem',
        'SSL_KEYFILE': 'certs/key.pem',
        'CACHE_TTL_TRADING': '5',
        'CACHE_TTL_RISK': '10',
        'CACHE_TTL_OVERVIEW': '30',
    }
    
    for key, default_value in defaults.items():
        if key not in existing_env:
            existing_env[key] = default_value
    
    # Write updated .env
    print("\nℹ️  Updating .env file...")
    from datetime import datetime
    with open(env_file, 'w', encoding='utf-8') as f:
        f.write("# Quantum Trader v2.0 - Configuration\n")
        f.write(f"# Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for key, value in sorted(existing_env.items()):
            f.write(f"{key}={value}\n")
    
    print(f"✅ Environment configured: {env_file}")
    print("\n⚠️  IMPORTANT: Review .env file and update passwords before production!")
    return True


def validate_installation():
    """Validate that all components are installed correctly."""
    print("\n🔍 Validating installation...")
    print("=" * 60)
    
    all_valid = True
    
    # Check Python packages
    packages = ['jose', 'passlib', 'redis']
    for package in packages:
        try:
            __import__(package)
            print(f"✅ {package}: installed")
        except ImportError:
            print(f"❌ {package}: NOT FOUND")
            all_valid = False
    
    # Check SSL certificates
    cert_file = Path("certs/cert.pem")
    key_file = Path("certs/key.pem")
    if cert_file.exists() and key_file.exists():
        print(f"✅ SSL certificates: present")
    else:
        print(f"⚠️  SSL certificates: NOT FOUND (run with --generate-cert)")
        all_valid = False
    
    # Check .env configuration
    env_file = Path(".env")
    if env_file.exists():
        print(f"✅ .env file: configured")
        
        # Check critical keys
        with open(env_file, 'r') as f:
            env_content = f.read()
            
        if 'JWT_SECRET_KEY' in env_content:
            print("   ├─ JWT_SECRET_KEY: present")
        else:
            print("   ├─ JWT_SECRET_KEY: MISSING")
            all_valid = False
        
        if 'API_KEY_ADMIN' in env_content:
            print("   ├─ API_KEY_ADMIN: present")
        else:
            print("   ├─ API_KEY_ADMIN: MISSING")
            all_valid = False
        
        if 'REDIS_URL' in env_content:
            print("   └─ REDIS_URL: present")
        else:
            print("   └─ REDIS_URL: MISSING")
            all_valid = False
    else:
        print(f"❌ .env file: NOT FOUND")
        all_valid = False
    
    # Check Redis connectivity
    try:
        import redis
        r = redis.Redis.from_url(os.getenv('REDIS_URL', 'redis://localhost:6379'))
        r.ping()
        print("✅ Redis: CONNECTED")
    except Exception as e:
        print(f"⚠️  Redis: NOT AVAILABLE ({e})")
        print("   Start Redis: docker run -d -p 6379:6379 redis:latest")
        all_valid = False
    
    print("\n" + "=" * 60)
    if all_valid:
        print("✅ Installation validated successfully!")
    else:
        print("⚠️  Some components need attention (see above)")
    
    return all_valid


def print_next_steps():
    """Print next steps after setup."""
    steps = """
╔══════════════════════════════════════════════════════════════╗
║                      NEXT STEPS                              ║
╚══════════════════════════════════════════════════════════════╝

1. Start Redis (if not running):
   docker run -d -p 6379:6379 redis:latest

2. Review .env file and update:
   - ADMIN_PASSWORD (if using user database)
   - API keys for production
   - FORCE_HTTPS=true (when ready for production)

3. Test HTTPS (development):
   uvicorn backend.main:app --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem --port 8443
   
   Then visit: https://localhost:8443/api/docs

4. Test Authentication:
   - Go to /api/docs
   - Try /api/auth/login endpoint
   - Use JWT token in "Authorize" button

5. Verify Caching:
   - Check response headers for X-Cache: HIT/MISS
   - Monitor Redis with: redis-cli MONITOR

6. Run Security Audit:
   python scripts/test_security.py

7. Run Performance Benchmarks:
   python scripts/test_performance.py

8. Production Deployment:
   - Use nginx reverse proxy (see backend/https_config.py)
   - Get Let's Encrypt certificate
   - Set FORCE_HTTPS=true
   - Enable rate limiting
   - Configure firewall

╔══════════════════════════════════════════════════════════════╗
║  For production setup instructions, see:                     ║
║  - backend/https_config.py (HTTPS setup)                     ║
║  - .env.security (configuration reference)                   ║
║  - FINAL_QA_REPORT.md (production checklist)                 ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(steps)


def main():
    """Main setup function."""
    parser = argparse.ArgumentParser(description='Production readiness setup for Quantum Trader')
    parser.add_argument('--generate-cert', action='store_true', help='Generate SSL certificates')
    parser.add_argument('--install-deps', action='store_true', help='Install dependencies')
    parser.add_argument('--configure-env', action='store_true', help='Configure environment')
    parser.add_argument('--validate', action='store_true', help='Validate installation only')
    parser.add_argument('--all', action='store_true', help='Run all setup steps')
    
    args = parser.parse_args()
    
    # If no args, run all
    if not any(vars(args).values()):
        args.all = True
    
    print_banner()
    
    success = True
    
    if args.all or args.install_deps:
        if not install_dependencies():
            success = False
    
    if args.all or args.generate_cert:
        if not generate_ssl_certificate():
            success = False
    
    if args.all or args.configure_env:
        if not configure_environment():
            success = False
    
    if args.all or args.validate:
        if not validate_installation():
            success = False
    
    if success:
        print_next_steps()
        print("\n✅ Setup completed successfully!")
        return 0
    else:
        print("\n⚠️  Setup completed with warnings. Please review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
