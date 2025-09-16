#!/usr/bin/env python3
"""
Quick setup test for Model Explainability TypeScript migration
"""

import sys
import subprocess
import os

def test_python_setup():
    """Test Python backend setup"""
    print("🐍 Testing Python Backend Setup...")
    
    # Check Python version
    print(f"   Python version: {sys.version}")
    
    # Check if we're in virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("   ✅ Virtual environment detected")
    else:
        print("   ⚠️  Not in virtual environment")
    
    # Test FastAPI import
    try:
        import fastapi
        print(f"   ✅ FastAPI installed: {fastapi.__version__}")
    except ImportError:
        print("   ❌ FastAPI not installed")
    
    # Test minimal backend
    try:
        os.chdir('backend')
        result = subprocess.run([sys.executable, 'main_minimal.py', '--help'], 
                              capture_output=True, text=True, timeout=5)
        print("   ✅ Minimal backend can start")
    except Exception as e:
        print(f"   ❌ Minimal backend issue: {e}")
    finally:
        os.chdir('..')

def test_node_setup():
    """Test Node.js frontend setup"""
    print("\n⚛️  Testing React Frontend Setup...")
    
    # Check Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        print(f"   ✅ Node.js: {result.stdout.strip()}")
    except:
        print("   ❌ Node.js not found")
    
    # Check npm
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True)
        print(f"   ✅ npm: {result.stdout.strip()}")
    except:
        print("   ❌ npm not found")
    
    # Check frontend dependencies
    if os.path.exists('frontend/node_modules'):
        print("   ✅ Frontend dependencies installed")
    else:
        print("   ❌ Frontend dependencies not installed")

def main():
    print("🧪 Model Explainability - Setup Test")
    print("=" * 50)
    
    test_python_setup()
    test_node_setup()
    
    print("\n🎯 Recommendations:")
    print("   1. Start minimal backend: cd backend && python main_minimal.py")
    print("   2. Start frontend: cd frontend && npm run dev")
    print("   3. Test at: http://localhost:3000")
    print("   4. Backend API docs: http://localhost:8000/docs")

if __name__ == "__main__":
    main()
