#!/usr/bin/env python3
"""
Fix JWT auth in all router files
"""
import os
import re

routers_dir = "quantumfond_backend/routers"
files_to_fix = [
    "ai_router.py",
    "admin_router.py", 
    "incident_router.py",
    "performance_router.py",
    "risk_router.py",
    "strategy_router.py",
    "system_router.py",
    "trades_router.py"
]

for filename in files_to_fix:
    filepath = os.path.join(routers_dir, filename)
    
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Remove fastapi_jwt_auth import
    content = re.sub(r'from fastapi_jwt_auth import AuthJWT\n', '', content)
    
    # Add verify_token import if not present
    if 'from .auth_router import' not in content:
        # Find the last import line
        imports = []
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if line.startswith('from ') or line.startswith('import '):
                imports.append(i)
        
        if imports:
            last_import = imports[-1]
            lines.insert(last_import + 1, 'from .auth_router import verify_token')
            content = '\n'.join(lines)
    
    # Remove Authorize: AuthJWT = Depends()
    content = re.sub(r',\s*Authorize:\s*AuthJWT\s*=\s*Depends\(\)', '', content)
    content = re.sub(r'Authorize:\s*AuthJWT\s*=\s*Depends\(\),?\s*', '', content)
    
    # Remove Authorize.jwt_required() calls
    content = re.sub(r'\s+Authorize\.jwt_required\(\)\s*\n', '\n', content)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"Fixed {filename}")

print("\nDone! All router files updated.")
