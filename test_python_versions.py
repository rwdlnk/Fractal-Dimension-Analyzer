#!/usr/bin/env python3
"""
Python Version Compatibility Test for Fractal Analyzer
Tests core functionality across different Python versions
"""

import sys
import warnings
from pathlib import Path

def test_python_version():
    """Test Python version compatibility."""
    version = sys.version_info
    print(f"Testing Python {version.major}.{version.minor}.{version.micro}")
    
    # Check minimum version
    if version < (3, 8):
        print("❌ ERROR: Python 3.8+ required")
        return False
    elif version < (3, 9):
        print("⚠️  WARNING: Python 3.9+ recommended for best performance")
    else:
        print("✅ Python version compatible")
    
    return True

def test_imports():
    """Test all required imports."""
    print("\nTesting imports...")
    
    required_modules = [
        ('numpy', 'np'),
        ('scipy', None),
        ('matplotlib.pyplot', 'plt'),
        ('numba', None),
    ]
    
    optional_modules = [
        ('pandas', 'pd'),
        ('seaborn', 'sns'),
        ('h5py', None),
    ]
    
    failed_imports = []
    
    # Test required modules
    for module, alias in required_modules:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    # Test optional modules
    print("\nOptional modules:")
    for module, alias in optional_modules:
        try:
            if alias:
                exec(f"import {module} as {alias}")
            else:
                exec(f"import {module}")
            print(f"✅ {module}")
        except ImportError:
            print(f"⚠️  {module} (optional)")
    
    return len(failed_imports) == 0

def test_numba_functionality():
    """Test numba JIT compilation."""
    print("\nTesting numba functionality...")
    
    try:
        from numba import jit
        import numpy as np
        
        @jit(nopython=True)
        def test_function(x):
            return x * 2 + 1
        
        # Test compilation and execution
        result = test_function(5.0)
        expected = 11.0
        
        if abs(result - expected) < 1e-10:
            print("✅ Numba JIT compilation working")
            return True
        else:
            print(f"❌ Numba computation error: {result} != {expected}")
            return False
            
    except Exception as e:
        print(f"❌ Numba error: {e}")
        return False

def test_fractal_analyzer_import():
    """Test fractal analyzer import and basic functionality."""
    print("\nTesting fractal_analyzer import...")
    
    try:
        # Add current directory to path if needed
        current_dir = Path(__file__).parent
        if current_dir not in sys.path:
            sys.path.insert(0, str(current_dir))
        
        # Test import
        from fractal_analyzer import FractalAnalyzer
        print("✅ FractalAnalyzer import successful")
        
        # Test basic instantiation
        analyzer = FractalAnalyzer(log_level='WARNING')  # Reduce log noise
        print("✅ FractalAnalyzer instantiation successful")
        
        # Test basic functionality
        if hasattr(analyzer, 'generate_fractal'):
            print("✅ Core methods available")
            return True
        else:
            print("❌ Core methods missing")
            return False
            
    except Exception as e:
        print(f"❌ FractalAnalyzer error: {e}")
        return False

def test_basic_fractal_generation():
    """Test basic fractal generation."""
    print("\nTesting basic fractal generation...")
    
    try:
        from fractal_analyzer import FractalAnalyzer
        
        # Suppress matplotlib warnings for testing
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            
            analyzer = FractalAnalyzer(log_level='ERROR')  # Minimal logging
            
            # Test Koch curve generation (small level)
            points, segments = analyzer.generate_fractal('koch', 2)
            
            if len(segments) > 0:
                print(f"✅ Koch curve generation: {len(segments)} segments")
                return True
            else:
                print("❌ No segments generated")
                return False
                
    except Exception as e:
        print(f"❌ Fractal generation error: {e}")
        return False

def run_all_tests():
    """Run comprehensive compatibility tests."""
    print("🔬 FRACTAL ANALYZER COMPATIBILITY TEST")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Module Imports", test_imports),
        ("Numba Functionality", test_numba_functionality),
        ("FractalAnalyzer Import", test_fractal_analyzer_import),
        ("Basic Generation", test_basic_fractal_generation),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print(f"\n{'=' * 50}")
    print("COMPATIBILITY TEST SUMMARY")
    print(f"{'=' * 50}")
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20s}: {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED - System ready for fractal analysis!")
        return True
    else:
        print("⚠️  Some tests failed - check installation")
        if passed >= 3:  # Core functionality works
            print("   Core functionality appears to work, may still be usable")
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
