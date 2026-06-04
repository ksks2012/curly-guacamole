#!/usr/bin/env python3
"""Quick syntax and import validation for refactored code."""

import sys
import traceback

def test_imports():
    """Test that all main modules can be imported."""
    modules_to_test = [
        'rag.client',
        'rag.code.indexer',
        'rag.knowledge.models',
        'ui.code_graph_tab',
        'utils.config',
    ]
    
    failed = []
    for module_name in modules_to_test:
        try:
            __import__(module_name)
            print(f"✓ {module_name}")
        except Exception as e:
            print(f"✗ {module_name}: {e}")
            failed.append((module_name, e))
    
    return len(failed) == 0, failed

def test_python_syntax():
    """Test Python syntax of key files."""
    import py_compile
    
    files_to_check = [
        'ui/code_graph_tab.py',
        'rag/client.py',
        'utils/file_processor.py',
    ]
    
    failed = []
    for file_path in files_to_check:
        try:
            py_compile.compile(file_path, doraise=True)
            print(f"✓ Syntax OK: {file_path}")
        except py_compile.PyCompileError as e:
            print(f"✗ Syntax error in {file_path}: {e}")
            failed.append((file_path, e))
    
    return len(failed) == 0, failed

if __name__ == "__main__":
    print("=" * 60)
    print("Testing Python Syntax")
    print("=" * 60)
    syntax_ok, syntax_errors = test_python_syntax()
    
    print("\n" + "=" * 60)
    print("Testing Module Imports")
    print("=" * 60)
    imports_ok, import_errors = test_imports()
    
    print("\n" + "=" * 60)
    if syntax_ok and imports_ok:
        print("✓ All tests passed!")
        sys.exit(0)
    else:
        print("✗ Some tests failed")
        if syntax_errors:
            print(f"  - {len(syntax_errors)} syntax error(s)")
        if import_errors:
            print(f"  - {len(import_errors)} import error(s)")
        sys.exit(1)
