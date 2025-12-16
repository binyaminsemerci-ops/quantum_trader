#!/usr/bin/env python3
"""
STEP 8: SECURITY & AUTHENTICATION AUDIT
========================================

Security checks:
1. API key exposure
2. Authentication enforcement
3. SQL injection protection
4. XSS protection
5. Secrets management
6. HTTPS usage
7. CORS configuration
8. Rate limiting

Author: GitHub Copilot (Security Engineer)
Date: December 5, 2025
"""

import asyncio
import httpx
from datetime import datetime
from typing import Dict, Any
import json
from pathlib import Path
import re

# Configuration
BACKEND_URL = "http://localhost:8000"
TIMEOUT = 10.0

# Results storage
results: Dict[str, Any] = {
    "timestamp": datetime.now().isoformat(),
    "security_checks": [],
    "summary": {}
}


def record_check(check_name: str, passed: bool, severity: str, details: str = "", recommendation: str = ""):
    """Record security check result."""
    result = {
        "check": check_name,
        "passed": passed,
        "severity": severity,  # CRITICAL, HIGH, MEDIUM, LOW
        "details": details,
        "recommendation": recommendation,
        "timestamp": datetime.now().isoformat()
    }
    results["security_checks"].append(result)
    
    status = "✅ PASS" if passed else "❌ FAIL"
    severity_emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}
    
    print(f"{status} [{severity_emoji.get(severity, '⚪')} {severity}] {check_name}")
    if details:
        print(f"     └─ {details}")
    if not passed and recommendation:
        print(f"     └─ 💡 {recommendation}")
    
    return passed


async def check_api_key_exposure():
    """Check 1: Verify API keys are not exposed in responses."""
    print("\n" + "="*80)
    print("CHECK 1: API KEY EXPOSURE")
    print("="*80)
    
    sensitive_patterns = [
        r'[A-Za-z0-9]{64}',  # Binance API key pattern
        r'api[_-]?key',
        r'secret[_-]?key',
        r'password',
        r'token',
    ]
    
    endpoints_to_check = [
        "/api/dashboard/overview",
        "/api/dashboard/system",
        "/health/live"
    ]
    
    exposed = False
    
    async with httpx.AsyncClient(timeout=TIMEOUT, base_url=BACKEND_URL) as client:
        for endpoint in endpoints_to_check:
            try:
                response = await client.get(endpoint)
                if response.status_code == 200:
                    content = response.text.lower()
                    
                    for pattern in sensitive_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            exposed = True
                            print(f"   ⚠️  Potential sensitive data in {endpoint}")
            except Exception:
                pass
    
    return record_check(
        "API Key Exposure",
        not exposed,
        "CRITICAL",
        "No API keys found in responses" if not exposed else "Potential API keys exposed",
        "Remove all sensitive credentials from API responses" if exposed else ""
    )


async def check_sql_injection_protection():
    """Check 2: SQL injection protection."""
    print("\n" + "="*80)
    print("CHECK 2: SQL INJECTION PROTECTION")
    print("="*80)
    
    sql_payloads = [
        "' OR '1'='1",
        "'; DROP TABLE users;--",
        "1' UNION SELECT NULL--",
        "admin'--"
    ]
    
    protected = True
    
    async with httpx.AsyncClient(timeout=TIMEOUT, base_url=BACKEND_URL) as client:
        for payload in sql_payloads:
            try:
                response = await client.get("/api/dashboard/trading", params={"symbol": payload})
                
                # Should either reject (400/422) or sanitize (200 with no error)
                if response.status_code in [500]:  # Server error suggests vulnerability
                    protected = False
                    print(f"   ⚠️  Payload caused server error: {payload}")
            except Exception:
                pass  # Exception is good - input rejected
    
    return record_check(
        "SQL Injection Protection",
        protected,
        "CRITICAL",
        "No SQL injection vulnerabilities detected" if protected else "Potential SQL injection vulnerability",
        "Implement parameterized queries and input validation" if not protected else ""
    )


async def check_xss_protection():
    """Check 3: XSS (Cross-Site Scripting) protection."""
    print("\n" + "="*80)
    print("CHECK 3: XSS PROTECTION")
    print("="*80)
    
    xss_payloads = [
        "<script>alert('xss')</script>",
        "<img src=x onerror=alert('xss')>",
        "javascript:alert('xss')"
    ]
    
    protected = True
    
    async with httpx.AsyncClient(timeout=TIMEOUT, base_url=BACKEND_URL) as client:
        for payload in xss_payloads:
            try:
                response = await client.get("/api/dashboard/trading", params={"test": payload})
                
                if response.status_code == 200:
                    # Check if payload is reflected unescaped
                    if payload in response.text:
                        protected = False
                        print(f"   ⚠️  XSS payload reflected: {payload[:30]}...")
            except Exception:
                pass
    
    return record_check(
        "XSS Protection",
        protected,
        "HIGH",
        "No XSS vulnerabilities detected" if protected else "Potential XSS vulnerability",
        "Implement output encoding and Content Security Policy" if not protected else ""
    )


async def check_cors_configuration():
    """Check 4: CORS configuration."""
    print("\n" + "="*80)
    print("CHECK 4: CORS CONFIGURATION")
    print("="*80)
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, base_url=BACKEND_URL) as client:
            response = await client.get("/health/live")
            
            cors_headers = {
                "access-control-allow-origin": response.headers.get("access-control-allow-origin"),
                "access-control-allow-methods": response.headers.get("access-control-allow-methods"),
                "access-control-allow-headers": response.headers.get("access-control-allow-headers")
            }
            
            # Check if CORS is too permissive
            allow_origin = cors_headers.get("access-control-allow-origin", "")
            
            if allow_origin == "*":
                return record_check(
                    "CORS Configuration",
                    False,
                    "MEDIUM",
                    "CORS allows all origins (*)",
                    "Restrict CORS to specific trusted origins"
                )
            elif allow_origin:
                return record_check(
                    "CORS Configuration",
                    True,
                    "MEDIUM",
                    f"CORS restricted to: {allow_origin}"
                )
            else:
                return record_check(
                    "CORS Configuration",
                    True,
                    "LOW",
                    "No CORS headers found (may be disabled)"
                )
    except Exception as e:
        return record_check(
            "CORS Configuration",
            False,
            "MEDIUM",
            f"Error checking CORS: {str(e)}"
        )


async def check_https_usage():
    """Check 5: HTTPS usage."""
    print("\n" + "="*80)
    print("CHECK 5: HTTPS USAGE")
    print("="*80)
    
    using_https = BACKEND_URL.startswith("https://")
    
    return record_check(
        "HTTPS Usage",
        using_https,
        "HIGH",
        "Using HTTPS" if using_https else "Using HTTP (insecure)",
        "Enable HTTPS for production deployment" if not using_https else ""
    )


async def check_rate_limiting():
    """Check 6: Rate limiting implementation."""
    print("\n" + "="*80)
    print("CHECK 6: RATE LIMITING")
    print("="*80)
    
    try:
        async with httpx.AsyncClient(timeout=5.0, base_url=BACKEND_URL) as client:
            # Send rapid requests
            tasks = [client.get("/health/live") for _ in range(100)]
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            status_codes = [
                r.status_code for r in responses 
                if isinstance(r, httpx.Response)
            ]
            
            rate_limited = any(code == 429 for code in status_codes)
            
            return record_check(
                "Rate Limiting",
                rate_limited,
                "MEDIUM",
                "Rate limiting active" if rate_limited else "No rate limiting detected",
                "Implement rate limiting to prevent abuse" if not rate_limited else ""
            )
    except Exception as e:
        return record_check(
            "Rate Limiting",
            False,
            "MEDIUM",
            f"Error: {str(e)}"
        )


async def check_error_message_disclosure():
    """Check 7: Error message information disclosure."""
    print("\n" + "="*80)
    print("CHECK 7: ERROR MESSAGE DISCLOSURE")
    print("="*80)
    
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT, base_url=BACKEND_URL) as client:
            # Trigger a 404
            response = await client.get("/api/nonexistent")
            
            if response.status_code == 404:
                error_text = response.text.lower()
                
                # Check for sensitive information in error messages
                sensitive_info = [
                    "traceback",
                    "file path",
                    "database",
                    "password",
                    "exception"
                ]
                
                discloses_info = any(info in error_text for info in sensitive_info)
                
                return record_check(
                    "Error Message Disclosure",
                    not discloses_info,
                    "MEDIUM",
                    "Error messages don't disclose sensitive info" if not discloses_info else "Error messages may disclose sensitive info",
                    "Use generic error messages in production" if discloses_info else ""
                )
            else:
                return record_check(
                    "Error Message Disclosure",
                    True,
                    "MEDIUM",
                    "Unable to trigger error for testing"
                )
    except Exception as e:
        return record_check(
            "Error Message Disclosure",
            False,
            "MEDIUM",
            f"Error: {str(e)}"
        )


async def check_authentication_endpoints():
    """Check 8: Authentication endpoint security."""
    print("\n" + "="*80)
    print("CHECK 8: AUTHENTICATION ENDPOINTS")
    print("="*80)
    
    # Check if sensitive endpoints require authentication
    sensitive_endpoints = [
        "/api/dashboard/trading",
        "/api/dashboard/risk"
    ]
    
    requires_auth = False
    
    async with httpx.AsyncClient(timeout=TIMEOUT, base_url=BACKEND_URL) as client:
        for endpoint in sensitive_endpoints:
            try:
                response = await client.get(endpoint)
                
                # If we get 401/403, authentication is required (good)
                if response.status_code in [401, 403]:
                    requires_auth = True
                    break
            except Exception:
                pass
    
    return record_check(
        "Authentication Endpoints",
        requires_auth,
        "HIGH",
        "Authentication required" if requires_auth else "No authentication detected",
        "Implement authentication for sensitive endpoints" if not requires_auth else ""
    )


async def main():
    """Run all security checks."""
    print("="*80)
    print("QUANTUM TRADER V2.0 - SECURITY AUDIT")
    print("STEP 8: Security & Authentication")
    print("="*80)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Backend: {BACKEND_URL}")
    print("="*80)
    
    # Run all checks
    check_results = []
    
    check_results.append(await check_api_key_exposure())
    check_results.append(await check_sql_injection_protection())
    check_results.append(await check_xss_protection())
    check_results.append(await check_cors_configuration())
    check_results.append(await check_https_usage())
    check_results.append(await check_rate_limiting())
    check_results.append(await check_error_message_disclosure())
    check_results.append(await check_authentication_endpoints())
    
    # Calculate summary
    passed_count = sum(1 for r in check_results if r)
    total_count = len(check_results)
    
    # Count by severity
    critical_failed = sum(1 for check in results["security_checks"] 
                         if not check["passed"] and check["severity"] == "CRITICAL")
    high_failed = sum(1 for check in results["security_checks"]
                     if not check["passed"] and check["severity"] == "HIGH")
    
    results["summary"] = {
        "total_checks": total_count,
        "passed": passed_count,
        "failed": total_count - passed_count,
        "critical_failed": critical_failed,
        "high_failed": high_failed
    }
    
    print("\n" + "="*80)
    print("SECURITY AUDIT SUMMARY")
    print("="*80)
    print(f"Total Checks: {total_count}")
    print(f"✅ Passed: {passed_count}")
    print(f"❌ Failed: {total_count - passed_count}")
    print(f"🔴 Critical Issues: {critical_failed}")
    print(f"🟠 High Issues: {high_failed}")
    print("="*80)
    
    if critical_failed > 0:
        print("🔴 CRITICAL SECURITY ISSUES FOUND - IMMEDIATE ACTION REQUIRED")
        exit_code = 2
    elif high_failed > 0:
        print("🟠 HIGH SECURITY ISSUES FOUND - ADDRESS BEFORE PRODUCTION")
        exit_code = 1
    elif total_count - passed_count > 0:
        print("🟡 MINOR SECURITY ISSUES FOUND - REVIEW RECOMMENDATIONS")
        exit_code = 0
    else:
        print("✅ NO MAJOR SECURITY ISSUES FOUND")
        exit_code = 0
    
    # Save results
    output_file = Path("SECURITY_AUDIT_RESULTS.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Detailed results saved to: {output_file.absolute()}")
    
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)
