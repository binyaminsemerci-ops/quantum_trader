#!/usr/bin/env python3
"""Mass update Binance API keys across all /etc/quantum/*.env files"""
import os
import glob
import sys

# The correct keys from apply-layer.env (the ones that work!)
CORRECT_API_KEY = "w2W60kzuCfPJKGIqSvmp0pqISUO8XKICjc5sD8QyJuJpp9LKQgvXKhtd09Ii3rwg"
CORRECT_API_SECRET = "QI18cg4zcbApc9uaDL8ZUmoAJQthQczZ9cKzORlSJfnK2zBEdLvSLb5ZEgZ6R1Kg"

def update_env_file(filepath):
    """Update API keys in a single .env file"""
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        
        updated = False
        new_lines = []
        
        for line in lines:
            original_line = line
            
            # Handle various key formats
            if line.startswith('BINANCE_API_KEY=') or line.startswith('BINANCE_TESTNET_API_KEY='):
                if '=' in line:
                    prefix = line.split('=')[0]
                    line = f"{prefix}={CORRECT_API_KEY}\n"
                    if line != original_line:
                        updated = True
            
            elif line.startswith('BINANCE_API_SECRET=') or line.startswith('BINANCE_TESTNET_API_SECRET='):
                if '=' in line:
                    prefix = line.split('=')[0]
                    line = f"{prefix}={CORRECT_API_SECRET}\n"
                    if line != original_line:
                        updated = True
            
            new_lines.append(line)
        
        if updated:
            with open(filepath, 'w') as f:
                f.writelines(new_lines)
            return True
        return False
    
    except Exception as e:
        print(f"❌ Error updating {filepath}: {e}")
        return False

def main():
    env_dir = "/etc/quantum"
    
    # Find all .env files
    env_files = glob.glob(f"{env_dir}/*.env")
    
    if not env_files:
        print(f"❌ No .env files found in {env_dir}")
        return 1
    
    print(f"Found {len(env_files)} .env files in {env_dir}")
    print(f"Updating to use keys from apply-layer.env...")
    print()
    
    updated_count = 0
    skipped_count = 0
    
    for filepath in sorted(env_files):
        filename = os.path.basename(filepath)
        
        if update_env_file(filepath):
            print(f"✅ Updated: {filename}")
            updated_count += 1
        else:
            print(f"⏭️  Skipped: {filename} (no keys found or already correct)")
            skipped_count += 1
    
    print()
    print(f"Summary:")
    print(f"  Updated: {updated_count} files")
    print(f"  Skipped: {skipped_count} files")
    print(f"  Total:   {len(env_files)} files")
    print()
    print("Next steps:")
    print("  1. Restart affected services: systemctl restart quantum-harvest-brain quantum-position-monitor")
    print("  2. Verify harvesting starts working")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
