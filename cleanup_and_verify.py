#!/usr/bin/env python3
"""
Project cleanup and verification script
"""

import os
import subprocess
import sys

def cleanup_project():
    """Clean up temporary and cache files"""
    print("🧹 Cleaning up project...")
    
    # Remove cache directories and files
    cache_patterns = [
        "__pycache__",
        "*.pyc",
        "*.pyo", 
        ".pytest_cache",
        ".coverage",
        "htmlcov",
        ".ipynb_checkpoints",
        "*.log"
    ]
    
    for pattern in cache_patterns:
        try:
            if pattern.startswith("*."):
                # Remove files with specific extensions
                subprocess.run(f"find . -name '{pattern}' -delete", shell=True, check=False)
            else:
                # Remove directories
                subprocess.run(f"find . -name '{pattern}' -type d -exec rm -rf {{}} + 2>/dev/null", shell=True, check=False)
            print(f"✅ Cleaned {pattern}")
        except:
            pass

def verify_project():
    """Verify project structure and functionality"""
    print("\n🔍 Verifying project structure...")
    
    # Check essential files
    essential_files = [
        "app.py",
        "requirements.txt", 
        "README.md",
        "USER_GUIDE.md",
        "DEPLOYMENT.md",
        "models/forecasting_models.py",
        "utils/data_processor.py",
        "utils/model_evaluator.py",
        "utils/visualization.py",
        "sample_kpi_data.xlsx"
    ]
    
    missing_files = []
    for file in essential_files:
        if os.path.exists(file):
            print(f"✅ {file}")
        else:
            print(f"❌ {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n⚠️ Missing files: {', '.join(missing_files)}")
    else:
        print("\n🎉 All essential files present!")
    
    return len(missing_files) == 0

def check_imports():
    """Test if all imports work correctly"""
    print("\n📦 Testing imports...")
    
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        import plotly.express as px
        import sklearn
        print("✅ All core dependencies available")
        
        # Test custom imports
        sys.path.append('.')
        from utils.data_processor import DataProcessor
        from utils.visualization import DataVisualizer 
        from models.forecasting_models import ForecastingEngine
        from utils.model_evaluator import ModelEvaluator
        print("✅ All custom modules importable")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main cleanup and verification"""
    print("=" * 60)
    print("🚀 KPI Forecasting Project - Cleanup & Verification")
    print("=" * 60)
    
    # Change to project directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Cleanup
    cleanup_project()
    
    # Verify
    structure_ok = verify_project()
    imports_ok = check_imports()
    
    print("\n" + "=" * 60)
    if structure_ok and imports_ok:
        print("🎉 PROJECT READY!")
        print("✅ All files present")
        print("✅ All imports working")
        print("✅ Cache files cleaned")
        print("\n🚀 To start the application:")
        print("   streamlit run app.py")
    else:
        print("⚠️ PROJECT ISSUES DETECTED")
        print("Please resolve the issues above before using the application")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
