#!/usr/bin/env python3
"""
Test script to verify the ST-GCN project setup
"""

import os
import sys
import importlib.util

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    required_modules = [
        'torch', 'numpy', 'pandas', 'sklearn', 
        'matplotlib', 'seaborn', 'tqdm'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✅ {module}")
        except ImportError:
            print(f"❌ {module}")
            missing_modules.append(module)
    
    if missing_modules:
        print(f"\nMissing modules: {missing_modules}")
        print("Please install requirements: pip install -r requirements.txt")
        return False
    
    return True

def test_project_structure():
    """Test if project structure is correct"""
    print("\nTesting project structure...")
    
    required_dirs = [
        'data/raw',
        'data/processed',
        'models',
        'src',
        'results'
    ]
    
    required_files = [
        'README.md',
        'requirements.txt',
        'src/data_processor.py',
        'src/stgcn_model.py',
        'src/train.py',
        'src/evaluate.py'
    ]
    
    # Check directories
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory: {dir_path}")
        else:
            print(f"❌ Directory: {dir_path}")
            return False
    
    # Check files
    for file_path in required_files:
        if os.path.exists(file_path):
            print(f"✅ File: {file_path}")
        else:
            print(f"❌ File: {file_path}")
            return False
    
    return True

def test_module_imports():
    """Test if local modules can be imported"""
    print("\nTesting local module imports...")
    
    # Add src to path
    src_path = os.path.join(os.getcwd(), 'src')
    if src_path not in sys.path:
        sys.path.insert(0, src_path)
    
    modules_to_test = [
        'data_processor',
        'stgcn_model',
        'train',
        'evaluate'
    ]
    
    for module_name in modules_to_test:
        try:
            module = importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except Exception as e:
            print(f"❌ {module_name}: {e}")
            return False
    
    return True

def test_data_files():
    """Check if data files are present"""
    print("\nChecking for data files...")
    
    train_csv = "data/raw/train.csv"
    val_csv = "data/raw/val.csv"
    
    if os.path.exists(train_csv):
        print(f"✅ Found: {train_csv}")
        train_exists = True
    else:
        print(f"ℹ️  Not found: {train_csv} (place your training data here)")
        train_exists = False
    
    if os.path.exists(val_csv):
        print(f"✅ Found: {val_csv}")
        val_exists = True
    else:
        print(f"ℹ️  Not found: {val_csv} (place your validation data here)")
        val_exists = False
    
    return train_exists, val_exists

def main():
    """Run all tests"""
    print("ST-GCN Project Setup Test")
    print("=" * 50)
    
    # Change to project directory
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    print(f"Working directory: {os.getcwd()}")
    
    all_passed = True
    
    # Test imports
    if not test_imports():
        all_passed = False
    
    # Test project structure
    if not test_project_structure():
        all_passed = False
    
    # Test module imports
    if not test_module_imports():
        all_passed = False
    
    # Check data files
    train_exists, val_exists = test_data_files()
    data_ready = train_exists and val_exists
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 All tests passed! Setup is correct.")
        if data_ready:
            print("✅ Data files found. You can run: python run_pipeline.py")
        else:
            print("ℹ️  Place your train.csv and val.csv in data/raw/ to proceed.")
    else:
        print("❌ Some tests failed. Please fix the issues above.")
    
    print("\nNext steps:")
    print("1. Install requirements: pip install -r requirements.txt")
    print("2. Place train.csv and val.csv in data/raw/")
    print("3. Run complete pipeline: python run_pipeline.py")
    print("   OR run individual scripts:")
    print("   - python src/data_processor.py")
    print("   - python src/train.py")
    print("   - python src/evaluate.py")

if __name__ == "__main__":
    main()