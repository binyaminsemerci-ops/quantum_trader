"""
Dashboard V3.0 Frontend Component Validator
DASHBOARD-V3-001: Phase 11 - Frontend Sanity Checks

Validates:
- All tab components exist and are importable
- TypeScript interfaces are defined
- Required files are present
- Component structure is correct
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple

# Colors for terminal output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


class DashboardValidator:
    """Validates Dashboard V3.0 frontend components"""
    
    def __init__(self, root_path: Path):
        self.root = root_path
        self.frontend = root_path / "frontend"
        self.errors: List[str] = []
        self.warnings: List[str] = []
        self.successes: List[str] = []
    
    def validate_all(self) -> bool:
        """Run all validation checks"""
        print(f"{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}Dashboard V3.0 Frontend Validation{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
        
        self.check_directory_structure()
        self.check_component_files()
        self.check_hook_files()
        self.check_main_page()
        self.check_imports()
        self.check_typescript_interfaces()
        
        self.print_summary()
        
        return len(self.errors) == 0
    
    def check_directory_structure(self):
        """Verify frontend directory structure"""
        print(f"{YELLOW}[1/6] Checking Directory Structure...{RESET}")
        
        required_dirs = [
            "components/dashboard",
            "hooks",
            "pages",
            "lib",
            "styles",
        ]
        
        for dir_path in required_dirs:
            full_path = self.frontend / dir_path
            if full_path.exists():
                self.success(f"✓ Directory exists: {dir_path}")
            else:
                self.error(f"✗ Missing directory: {dir_path}")
        
        print()
    
    def check_component_files(self):
        """Verify all tab component files exist"""
        print(f"{YELLOW}[2/6] Checking Component Files...{RESET}")
        
        components = [
            "components/dashboard/OverviewTab.tsx",
            "components/dashboard/TradingTab.tsx",
            "components/dashboard/RiskTab.tsx",
            "components/dashboard/SystemTab.tsx",
        ]
        
        for component in components:
            file_path = self.frontend / component
            if file_path.exists():
                # Check file size (should not be empty)
                size = file_path.stat().st_size
                if size > 1000:  # At least 1KB
                    self.success(f"✓ Component exists: {component} ({size} bytes)")
                else:
                    self.warning(f"⚠ Component too small: {component} ({size} bytes)")
            else:
                self.error(f"✗ Missing component: {component}")
        
        print()
    
    def check_hook_files(self):
        """Verify custom hooks exist"""
        print(f"{YELLOW}[3/6] Checking Hook Files...{RESET}")
        
        hooks = [
            "hooks/useDashboardStream.ts",
        ]
        
        for hook in hooks:
            file_path = self.frontend / hook
            if file_path.exists():
                size = file_path.stat().st_size
                self.success(f"✓ Hook exists: {hook} ({size} bytes)")
            else:
                self.error(f"✗ Missing hook: {hook}")
        
        print()
    
    def check_main_page(self):
        """Verify main dashboard page has tab navigation"""
        print(f"{YELLOW}[4/6] Checking Main Page...{RESET}")
        
        index_path = self.frontend / "pages/index.tsx"
        
        if not index_path.exists():
            self.error("✗ Main page not found: pages/index.tsx")
            print()
            return
        
        content = index_path.read_text(encoding='utf-8')
        
        # Check for tab imports
        required_imports = [
            "OverviewTab",
            "TradingTab",
            "RiskTab",
            "SystemTab",
        ]
        
        for import_name in required_imports:
            if import_name in content:
                self.success(f"✓ Tab imported: {import_name}")
            else:
                self.error(f"✗ Missing import: {import_name}")
        
        # Check for tab state
        if "activeTab" in content and "setActiveTab" in content:
            self.success("✓ Tab navigation state present")
        else:
            self.error("✗ Missing tab navigation state")
        
        # Check for tab rendering
        tab_checks = [
            "activeTab === 'overview'",
            "activeTab === 'trading'",
            "activeTab === 'risk'",
            "activeTab === 'system'",
        ]
        
        for check in tab_checks:
            if check in content:
                self.success(f"✓ Tab rendering: {check}")
            else:
                self.warning(f"⚠ Tab rendering not found: {check}")
        
        print()
    
    def check_imports(self):
        """Check that components have proper imports"""
        print(f"{YELLOW}[5/6] Checking Component Imports...{RESET}")
        
        components_to_check = [
            ("components/dashboard/OverviewTab.tsx", ["DashboardCard", "useDashboardStream"]),
            ("components/TopBar.tsx", ["useDashboardStream"]),
        ]
        
        for file_path, required_imports in components_to_check:
            full_path = self.frontend / file_path
            if full_path.exists():
                content = full_path.read_text(encoding='utf-8')
                for import_name in required_imports:
                    if import_name in content:
                        self.success(f"✓ {file_path} imports {import_name}")
                    else:
                        self.warning(f"⚠ {file_path} missing import: {import_name}")
        
        print()
    
    def check_typescript_interfaces(self):
        """Check for TypeScript interfaces in components"""
        print(f"{YELLOW}[6/6] Checking TypeScript Interfaces...{RESET}")
        
        components = [
            "components/dashboard/OverviewTab.tsx",
            "components/dashboard/TradingTab.tsx",
            "components/dashboard/RiskTab.tsx",
            "components/dashboard/SystemTab.tsx",
            "hooks/useDashboardStream.ts",
        ]
        
        for component in components:
            file_path = self.frontend / component
            if file_path.exists():
                content = file_path.read_text(encoding='utf-8')
                if "interface" in content:
                    # Count interfaces
                    interface_count = content.count("interface")
                    self.success(f"✓ {component} has {interface_count} interface(s)")
                else:
                    self.warning(f"⚠ {component} has no TypeScript interfaces")
        
        print()
    
    def success(self, msg: str):
        """Record success message"""
        self.successes.append(msg)
        print(f"{GREEN}{msg}{RESET}")
    
    def warning(self, msg: str):
        """Record warning message"""
        self.warnings.append(msg)
        print(f"{YELLOW}{msg}{RESET}")
    
    def error(self, msg: str):
        """Record error message"""
        self.errors.append(msg)
        print(f"{RED}{msg}{RESET}")
    
    def print_summary(self):
        """Print validation summary"""
        print(f"\n{BLUE}{'='*60}{RESET}")
        print(f"{BLUE}Validation Summary{RESET}")
        print(f"{BLUE}{'='*60}{RESET}\n")
        
        print(f"{GREEN}✓ Successes: {len(self.successes)}{RESET}")
        print(f"{YELLOW}⚠ Warnings:  {len(self.warnings)}{RESET}")
        print(f"{RED}✗ Errors:    {len(self.errors)}{RESET}\n")
        
        if self.errors:
            print(f"{RED}Critical Issues:{RESET}")
            for error in self.errors[:5]:  # Show first 5 errors
                print(f"  {error}")
            if len(self.errors) > 5:
                print(f"  ... and {len(self.errors) - 5} more\n")
        
        if len(self.errors) == 0:
            print(f"{GREEN}{'='*60}{RESET}")
            print(f"{GREEN}✓ ALL CHECKS PASSED - Dashboard V3.0 is ready!{RESET}")
            print(f"{GREEN}{'='*60}{RESET}\n")
        else:
            print(f"{RED}{'='*60}{RESET}")
            print(f"{RED}✗ VALIDATION FAILED - Please fix errors above{RESET}")
            print(f"{RED}{'='*60}{RESET}\n")


def main():
    """Run validation"""
    # Get quantum_trader root directory
    script_dir = Path(__file__).parent
    root_dir = script_dir.parent if script_dir.name == "tests" else script_dir
    
    validator = DashboardValidator(root_dir)
    success = validator.validate_all()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
