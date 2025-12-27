"""
HTTPS Configuration for Quantum Trader v2.0

Provides:
- SSL/TLS certificate setup
- HTTPS redirect middleware
- Security headers
- Production-ready configuration
"""

import os
from pathlib import Path
from typing import Optional

from fastapi import Request
from fastapi.responses import RedirectResponse
from starlette.middleware.base import BaseHTTPMiddleware


class HTTPSRedirectMiddleware(BaseHTTPMiddleware):
    """Redirect HTTP to HTTPS in production."""
    
    async def dispatch(self, request: Request, call_next):
        # Only redirect if HTTPS is enabled
        if os.getenv("FORCE_HTTPS", "false").lower() == "true":
            if request.url.scheme == "http":
                url = request.url.replace(scheme="https")
                return RedirectResponse(url=str(url), status_code=301)
        
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses."""
    
    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self' data:; "
            "connect-src 'self' https://testnet.binance.vision;"
        )
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        return response


def get_ssl_config() -> dict:
    """Get SSL configuration for uvicorn."""
    ssl_keyfile = os.getenv("SSL_KEYFILE", "certs/key.pem")
    ssl_certfile = os.getenv("SSL_CERTFILE", "certs/cert.pem")
    
    # Check if certificates exist
    if Path(ssl_keyfile).exists() and Path(ssl_certfile).exists():
        return {
            "ssl_keyfile": ssl_keyfile,
            "ssl_certfile": ssl_certfile,
        }
    
    return {}


# Certificate generation script (for development only)
def generate_self_signed_cert():
    """Generate self-signed certificate for development."""
    import subprocess
    from pathlib import Path
    
    cert_dir = Path("certs")
    cert_dir.mkdir(exist_ok=True)
    
    key_file = cert_dir / "key.pem"
    cert_file = cert_dir / "cert.pem"
    
    if key_file.exists() and cert_file.exists():
        print("✅ SSL certificates already exist")
        return
    
    print("🔐 Generating self-signed SSL certificate for development...")
    
    try:
        # Generate private key and certificate
        subprocess.run([
            "openssl", "req", "-x509", "-newkey", "rsa:4096",
            "-keyout", str(key_file),
            "-out", str(cert_file),
            "-days", "365",
            "-nodes",
            "-subj", "/CN=localhost"
        ], check=True)
        
        print(f"✅ Certificate created: {cert_file}")
        print(f"✅ Private key created: {key_file}")
        print("⚠️  Note: This is a self-signed certificate for development only!")
        print("   For production, use Let's Encrypt or a proper CA.")
    except FileNotFoundError:
        print("❌ OpenSSL not found. Install OpenSSL to generate certificates.")
        print("   On Windows: https://slproweb.com/products/Win32OpenSSL.html")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to generate certificate: {e}")


# Production HTTPS setup instructions
HTTPS_SETUP_INSTRUCTIONS = """
╔════════════════════════════════════════════════════════════════╗
║                 HTTPS SETUP INSTRUCTIONS                       ║
╚════════════════════════════════════════════════════════════════╝

FOR DEVELOPMENT (Self-Signed):
------------------------------
1. Run: python -c "from backend.https_config import generate_self_signed_cert; generate_self_signed_cert()"
2. Set environment: FORCE_HTTPS=false
3. Start server with: uvicorn backend.main:app --ssl-keyfile certs/key.pem --ssl-certfile certs/cert.pem --port 8443

FOR PRODUCTION (Let's Encrypt):
-------------------------------
1. Install certbot:
   - Ubuntu/Debian: sudo apt install certbot
   - macOS: brew install certbot

2. Get certificate:
   sudo certbot certonly --standalone -d yourdomain.com

3. Certificates will be at:
   - /etc/letsencrypt/live/yourdomain.com/fullchain.pem
   - /etc/letsencrypt/live/yourdomain.com/privkey.pem

4. Set environment variables:
   export SSL_CERTFILE=/etc/letsencrypt/live/yourdomain.com/fullchain.pem
   export SSL_KEYFILE=/etc/letsencrypt/live/yourdomain.com/privkey.pem
   export FORCE_HTTPS=true

5. Start with SSL:
   uvicorn backend.main:app --host 0.0.0.0 --port 443 \\
     --ssl-keyfile $SSL_KEYFILE --ssl-certfile $SSL_CERTFILE

6. Setup auto-renewal:
   sudo certbot renew --dry-run
   (Add to crontab: 0 0 * * * certbot renew --quiet)

USING NGINX REVERSE PROXY (Recommended):
----------------------------------------
1. Install nginx: sudo apt install nginx

2. Configure /etc/nginx/sites-available/quantum-trader:

server {
    listen 80;
    server_name yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;
    
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    
    # Modern SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;
    
    # HSTS
    add_header Strict-Transport-Security "max-age=31536000" always;
    
    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

3. Enable site:
   sudo ln -s /etc/nginx/sites-available/quantum-trader /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl restart nginx

4. Start backend (HTTP on localhost):
   uvicorn backend.main:app --host 127.0.0.1 --port 8000

DOCKER WITH HTTPS:
-----------------
See docker-compose.https.yml for complete setup with nginx + certbot.

╚════════════════════════════════════════════════════════════════╝
"""


def print_https_setup():
    """Print HTTPS setup instructions."""
    print(HTTPS_SETUP_INSTRUCTIONS)


if __name__ == "__main__":
    print_https_setup()
    
    # Generate dev certificate if requested
    import sys
    if "--generate-cert" in sys.argv:
        generate_self_signed_cert()
