"""
Fix Unicode Emoji Issue - Windows Compatibility

Problem: Windows console (cp1252) cannot encode Unicode emojis used in logger.info() calls.
Solution: Replace all emoji characters with ASCII equivalents.

This script will:
1. Find all Python files with emoji characters in logging statements
2. Replace emojis with ASCII equivalents
3. Create backup of original files
4. Generate a log of all changes
"""

import os
import re
from pathlib import Path
from datetime import datetime

# Emoji mappings to ASCII
EMOJI_REPLACEMENTS = {
    # Basic symbols
    '\u2705': '[OK]',  # [OK] check mark
    '\U0001f680': '[ROCKET]',  # [ROCKET] rocket
    '\U0001f50d': '[SEARCH]',  # [SEARCH] magnifying glass
    '\U0001f4e1': '[SIGNAL]',  # [SIGNAL] satellite antenna
    '\U0001f4ca': '[CHART]',  # [CHART] bar chart
    '\U0001f4cb': '[CLIPBOARD]',  # [CLIPBOARD] clipboard
    '\U0001f6ab': '[BLOCKED]',  # [BLOCKED] no entry sign
    '\U0001f3af': '[TARGET]',  # [TARGET] direct hit
    '\U0001f534': '[RED_CIRCLE]',  # [RED_CIRCLE] red circle
    '\U0001f7e2': '[GREEN_CIRCLE]',  # [GREEN_CIRCLE] green circle
    '\u23ed\ufe0f': '[SKIP]',  # [SKIP] next track
    '\U0001f4b0': '[MONEY]',  # [MONEY] money bag
    '\U0001f4bc': '[BRIEFCASE]',  # [BRIEFCASE] briefcase
    '\U0001f4dd': '[MEMO]',  # [MEMO] memo
    '\U0001f3c1': '[CHECKERED_FLAG]',  # [CHECKERED_FLAG] checkered flag
    '\U0001f4c8': '[CHART_UP]',  # [CHART_UP] chart increasing
    '\U0001f9ea': '[TEST_TUBE]',  # [TEST_TUBE] test tube
    '\u26a0\ufe0f': '[WARNING]',  # [WARNING] warning
    '\U0001f6e1\ufe0f': '[SHIELD]',  # [SHIELD] shield
    '\U0001f6a8': '[ALERT]',  # [ALERT] police car light
    '\U0001f441\ufe0f': '[EYE]',  # [EYE] eye
}

def fix_file(filepath: Path, log_file):
    """Fix emoji issues in a single file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        changes_made = False
        changes = []
        
        # Replace each emoji
        for emoji, replacement in EMOJI_REPLACEMENTS.items():
            if emoji in content:
                count = content.count(emoji)
                content = content.replace(emoji, replacement)
                changes_made = True
                changes.append(f"  - Replaced {count}x {repr(emoji)} with {replacement}")
        
        if changes_made:
            # Create backup
            backup_path = filepath.with_suffix(filepath.suffix + '.emoji_backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Write fixed content
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            
            # Log changes
            log_msg = f"\n✓ FIXED: {filepath.relative_to(Path.cwd())}\n" + "\n".join(changes)
            print(log_msg)
            log_file.write(log_msg + "\n")
            
            return True, len(changes)
        
        return False, 0
            
    except Exception as e:
        error_msg = f"\n✗ ERROR: {filepath}: {e}"
        print(error_msg)
        log_file.write(error_msg + "\n")
        return False, 0

def main():
    """Main function to fix all Python files"""
    root_dir = Path.cwd()
    log_path = root_dir / "EMOJI_FIX_LOG.md"
    
    # Find all Python files
    python_files = list(root_dir.rglob("*.py"))
    
    # Filter out virtual environments and hidden directories
    python_files = [
        f for f in python_files 
        if not any(part.startswith('.') or part == 'venv' or part == '__pycache__' 
                   for part in f.parts)
    ]
    
    print(f"\n{'='*70}")
    print(f"FIXING UNICODE EMOJI ISSUES - Windows Compatibility")
    print(f"{'='*70}\n")
    print(f"Found {len(python_files)} Python files to scan\n")
    
    files_fixed = 0
    total_replacements = 0
    
    with open(log_path, 'w', encoding='utf-8') as log_file:
        log_file.write(f"# Unicode Emoji Fix Log\n\n")
        log_file.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        log_file.write(f"**Problem:** Windows console (cp1252) cannot encode Unicode emojis\n\n")
        log_file.write(f"**Solution:** Replaced all emojis with ASCII equivalents\n\n")
        log_file.write(f"## Files Modified\n\n")
        
        for py_file in python_files:
            fixed, count = fix_file(py_file, log_file)
            if fixed:
                files_fixed += 1
                total_replacements += count
    
    print(f"\n{'='*70}")
    print(f"FIX COMPLETE!")
    print(f"{'='*70}\n")
    print(f"Files fixed: {files_fixed}")
    print(f"Total replacements: {total_replacements}")
    print(f"\nLog saved to: {log_path}")
    print(f"\nBackup files created with .emoji_backup extension")
    print(f"You can revert by running: git checkout HEAD -- <file>\n")

if __name__ == "__main__":
    main()
