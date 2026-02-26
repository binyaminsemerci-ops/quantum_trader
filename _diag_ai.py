#!/usr/bin/env python3
"""Check AI Engine endpoint that exit_manager calls, test it, find why it hangs"""
import urllib.request, urllib.error, json, subprocess, time, sys, re

print("=== AI ENGINE ENDPOINT DIAGNOSIS ===")

# 1. Find what URL exit_manager posts to
em_path = "microservices/autonomous_trader/exit_manager.py"
with open(em_path) as f:
    code = f.read()

# Find URL patterns
urls = re.findall(r'["\']([^"\']*(?:evaluate|exit_eval|predict|score|decision)[^"\']*)["\']', code, re.IGNORECASE)
post_lines = [(i+1, line.strip()) for i, line in enumerate(code.splitlines()) if '.post(' in line or '.get(' in line or 'await' in line and 'http' in line]

print(f"URL patterns found: {urls[:10]}")
print(f"HTTP call lines: {post_lines[:5]}")

# Find AI engine URL and endpoint
ai_url_match = re.search(r'self\.ai_engine_url\s*[+=]\s*["\']?([^"\';\n]+)', code)
if ai_url_match:
    print(f"ai_engine_url var assignment: {ai_url_match.group(1)}")

# Find actual await http call
for i, line in enumerate(code.splitlines()):
    if 'await' in line and ('post' in line or 'get' in line):
        print(f"  Line {i+1}: {line.strip()}")

# 2. Test the endpoints manually
base = "http://127.0.0.1:8001"
endpoints_to_test = [
    "/health",
    "/evaluate_exit",
    "/exit_eval",
    "/predict",
    "/signal",
    "/evaluate",
]

print("\n--- Testing AI Engine endpoints ---")
for ep in endpoints_to_test:
    url = base + ep
    try:
        req = urllib.request.Request(url, method='GET')
        with urllib.request.urlopen(req, timeout=3) as resp:
            body = resp.read()[:100]
            print(f"  GET {ep}: {resp.status} — {body}")
    except urllib.error.HTTPError as e:
        if e.code == 405:
            # Try POST
            try:
                req2 = urllib.request.Request(url, data=b'{}', method='POST',
                    headers={'Content-Type': 'application/json'})
                with urllib.request.urlopen(req2, timeout=3) as resp:
                    print(f"  POST {ep}: {resp.status}")
            except Exception as e2:
                print(f"  POST {ep}: {e2}")
        else:
            print(f"  GET {ep}: HTTP {e.code}")
    except Exception as e:
        print(f"  GET {ep}: {type(e).__name__}: {str(e)[:50]}")

# 3. Check what the AI Engine service actually exposes
r2 = subprocess.run(['curl', '-s', '--max-time', '3', f'{base}/openapi.json'],
                   capture_output=True, text=True)
if r2.returncode == 0 and r2.stdout.strip():
    try:
        api = json.loads(r2.stdout)
        paths = list(api.get('paths', {}).keys())
        print(f"\nAPI paths: {paths}")
    except:
        print(f"\nOpenAPI raw: {r2.stdout[:200]}")
else:
    print(f"\nOpenAPI: {r2.returncode} {r2.stderr[:50]}")

# 4. Check AI Engine logs for errors
r3 = subprocess.run(
    ['journalctl', '-u', 'quantum-ai-engine', '-n', '30', '--no-pager'],
    capture_output=True, text=True
)
# Find errors or the exit evaluation related lines
relevant = [l for l in r3.stdout.splitlines() 
            if any(w in l.lower() for w in ['error', 'warning', 'exit', 'evaluate', 'timeout', 'exception'])]
print(f"\nAI Engine relevant logs:")
for l in relevant[-10:]:
    print(f"  {l.split('] ')[-1][:100]}")

print("\n=== DONE ===")
