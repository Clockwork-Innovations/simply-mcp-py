#!/usr/bin/env python3
"""
Phase 1 Syntax & Imports Validation Script
Validates all Phase 1 code files for syntax errors and import resolution.
"""

import sys
import py_compile
import importlib.util
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple
import traceback

# Define all Phase 1 files to validate
PHASE1_FILES = {
    "HTTP Transport (Phase 1.1)": [
        "src/simply_mcp/transports/http_transport.py",
        "src/simply_mcp/core/auth.py",
        "src/simply_mcp/core/rate_limit.py",
        "src/simply_mcp/core/http_config.py",
        "src/simply_mcp/core/security.py",
        "src/simply_mcp/monitoring/http_metrics.py",
    ],
    "Async File Upload (Phase 1.2)": [
        "demo/gemini/upload_handler_foundation.py",
        "demo/gemini/upload_handler_feature.py",
        "demo/gemini/upload_handler_polish.py",
        "demo/gemini/http_server_with_uploads.py",
    ],
    "Session Storage (Phase 1.3)": [
        "demo/gemini/storage/base.py",
        "demo/gemini/storage/sqlite.py",
        "demo/gemini/storage/postgresql.py",
        "demo/gemini/storage/mongodb.py",
        "demo/gemini/storage/manager.py",
        "demo/gemini/storage/config.py",
        "demo/gemini/storage/migrations.py",
    ],
}

class ValidationReport:
    def __init__(self):
        self.syntax_passed = []
        self.syntax_failed = []
        self.import_passed = []
        self.import_failed = []
        self.subsystem_status = {}

    def add_syntax_pass(self, file_path: str):
        self.syntax_passed.append(file_path)

    def add_syntax_fail(self, file_path: str, error: str):
        self.syntax_failed.append((file_path, error))

    def add_import_pass(self, module_name: str):
        self.import_passed.append(module_name)

    def add_import_fail(self, module_name: str, error: str):
        self.import_failed.append((module_name, error))

    def print_report(self):
        print("\n" + "="*80)
        print("PHASE 1 SYNTAX & IMPORTS VALIDATION REPORT")
        print("="*80)

        # Compilation Results
        print("\n1. COMPILATION RESULTS")
        print("-" * 80)
        total_files = len(self.syntax_passed) + len(self.syntax_failed)
        print(f"   Total files checked: {total_files}")
        print(f"   Files compiled successfully: {len(self.syntax_passed)}")
        print(f"   Files with errors: {len(self.syntax_failed)}")

        if self.syntax_failed:
            print("\n   ERROR DETAILS:")
            for file_path, error in self.syntax_failed:
                print(f"\n   ❌ {file_path}")
                print(f"      {error}")
        else:
            print("\n   ✅ All files compiled successfully!")

        # Import Results
        print("\n2. IMPORT RESULTS")
        print("-" * 80)
        total_imports = len(self.import_passed) + len(self.import_failed)
        print(f"   Total modules checked: {total_imports}")
        print(f"   Imports resolved: {len(self.import_passed)}")
        print(f"   Failed imports: {len(self.import_failed)}")

        if self.import_failed:
            print("\n   IMPORT ERROR DETAILS:")
            for module, error in self.import_failed:
                print(f"\n   ❌ {module}")
                print(f"      {error}")
        else:
            print("\n   ✅ All imports resolved successfully!")

        # Status per Subsystem
        print("\n3. STATUS PER SUBSYSTEM")
        print("-" * 80)
        for subsystem, status in self.subsystem_status.items():
            status_icon = "✅ PASS" if status else "❌ FAIL"
            print(f"   {subsystem}: {status_icon}")

        # Overall Verdict
        print("\n4. OVERALL VERDICT")
        print("-" * 80)
        all_pass = len(self.syntax_failed) == 0 and len(self.import_failed) == 0

        if all_pass:
            print("   ✅ ALL PASS - Ready for unit tests")
        else:
            print("   ❌ FAILURES DETECTED")
            print("\n   ISSUES:")
            if self.syntax_failed:
                print(f"      - {len(self.syntax_failed)} file(s) with syntax errors")
                for file_path, _ in self.syntax_failed:
                    print(f"        • {file_path}")
            if self.import_failed:
                print(f"      - {len(self.import_failed)} module(s) with import errors")
                for module, _ in self.import_failed:
                    print(f"        • {module}")

        print("\n" + "="*80)

        return all_pass


def validate_syntax(file_path: str) -> Tuple[bool, str]:
    """Validate Python syntax by compiling the file."""
    try:
        py_compile.compile(file_path, doraise=True)
        return True, ""
    except py_compile.PyCompileError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"


def validate_import(file_path: str) -> Tuple[bool, str]:
    """Validate that a module can be imported."""
    # Convert file path to module name
    path = Path(file_path)

    # For demo files, we need to handle them specially
    if "demo/" in file_path:
        # Add demo directory to sys.path temporarily
        project_root = Path(__file__).parent
        demo_parent = project_root / "demo"
        gemini_dir = project_root / "demo" / "gemini"

        original_path = sys.path.copy()
        original_modules = set(sys.modules.keys())

        try:
            # Add both paths to handle different import styles
            if str(demo_parent) not in sys.path:
                sys.path.insert(0, str(demo_parent))
            if str(gemini_dir) not in sys.path:
                sys.path.insert(0, str(gemini_dir))

            # For storage modules with relative imports, import as a package
            if "demo/gemini/storage/" in file_path and file_path != "demo/gemini/storage/base.py":
                # Import as part of the package
                module_name = f"storage.{path.stem}"
                abs_path = project_root / file_path

                # First ensure the parent package exists
                parent_pkg = "storage"
                if parent_pkg not in sys.modules:
                    storage_init = project_root / "demo" / "gemini" / "storage" / "__init__.py"
                    pkg_spec = importlib.util.spec_from_file_location(parent_pkg, storage_init)
                    if pkg_spec and pkg_spec.loader:
                        pkg_module = importlib.util.module_from_spec(pkg_spec)
                        sys.modules[parent_pkg] = pkg_module
                        pkg_spec.loader.exec_module(pkg_module)

                # Now import the actual module
                spec = importlib.util.spec_from_file_location(module_name, abs_path, submodule_search_locations=[str(gemini_dir / "storage")])
            else:
                # Load module using spec
                abs_path = project_root / file_path
                module_name = path.stem
                spec = importlib.util.spec_from_file_location(module_name, abs_path)

            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                # Register in sys.modules temporarily
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                    return True, ""
                except Exception as e:
                    error_msg = str(e)
                    # Filter out acceptable errors for demo code
                    if "No module named 'google.generativeai'" in error_msg:
                        return True, "Skipped (optional dependency: google-generativeai)"
                    raise
            else:
                return False, "Could not create module spec"
        except Exception as e:
            error_msg = str(e)
            # Filter out acceptable errors for demo code
            if "No module named 'google.generativeai'" in error_msg:
                return True, "Skipped (optional dependency: google-generativeai)"
            return False, f"{type(e).__name__}: {error_msg}"
        finally:
            # Clean up new modules
            new_modules = set(sys.modules.keys()) - original_modules
            for mod in new_modules:
                if mod in sys.modules:
                    del sys.modules[mod]
            # Restore original path
            sys.path = original_path

    # For src files, use proper module import
    if file_path.startswith("src/"):
        module_path = file_path.replace("src/", "").replace(".py", "").replace("/", ".")
        try:
            spec = importlib.util.find_spec(module_path)
            if spec is None:
                return False, f"Module not found: {module_path}"

            module = importlib.import_module(module_path)
            return True, ""
        except Exception as e:
            return False, f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"

    return False, "Unknown module type"


def main():
    """Main validation function."""
    report = ValidationReport()
    project_root = Path(__file__).parent

    print("Starting Phase 1 Validation...")
    print(f"Project root: {project_root}")

    # Step 1: Syntax Validation
    print("\n" + "="*80)
    print("STEP 1: SYNTAX VALIDATION")
    print("="*80)

    for subsystem, files in PHASE1_FILES.items():
        print(f"\nValidating {subsystem}...")
        subsystem_passed = True

        for file_rel in files:
            file_path = project_root / file_rel
            print(f"  Checking: {file_rel}...", end=" ")

            if not file_path.exists():
                print(f"❌ FILE NOT FOUND")
                report.add_syntax_fail(file_rel, "File does not exist")
                subsystem_passed = False
                continue

            success, error = validate_syntax(str(file_path))
            if success:
                print("✅")
                report.add_syntax_pass(file_rel)
            else:
                print("❌")
                report.add_syntax_fail(file_rel, error)
                subsystem_passed = False

        report.subsystem_status[f"{subsystem} (Syntax)"] = subsystem_passed

    # Step 2: Import Validation
    print("\n" + "="*80)
    print("STEP 2: IMPORT VALIDATION")
    print("="*80)

    for subsystem, files in PHASE1_FILES.items():
        print(f"\nValidating imports for {subsystem}...")
        subsystem_passed = True

        for file_rel in files:
            file_path = project_root / file_rel
            print(f"  Importing: {file_rel}...", end=" ")

            if not file_path.exists():
                print("⏭️  SKIPPED (file not found)")
                continue

            success, error = validate_import(file_rel)
            if success:
                print("✅")
                report.add_import_pass(file_rel)
            else:
                print("❌")
                report.add_import_fail(file_rel, error)
                subsystem_passed = False

        report.subsystem_status[f"{subsystem} (Imports)"] = subsystem_passed

    # Step 3: Dependency Check
    print("\n" + "="*80)
    print("STEP 3: DEPENDENCY CHECK")
    print("="*80)

    try:
        result = subprocess.run(
            ["pip", "check"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            print("✅ All package dependencies satisfied")
        else:
            print("⚠️  Dependency issues detected:")
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
    except Exception as e:
        print(f"⚠️  Could not run pip check: {e}")

    # Generate and print final report
    all_pass = report.print_report()

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
