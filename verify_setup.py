"""
verify_setup.py
───────────────
Run this FIRST to confirm your environment is ready.

    python verify_setup.py

Checks:
    ✅ Python version >= 3.10
    ✅ All required packages installed
    ✅ Folder structure exists
    ✅ Data files (after generate_data.py)
"""

import sys
import importlib
from pathlib import Path

REQUIRED_PACKAGES = [
    ("pandas",          "pandas"),
    ("numpy",           "numpy"),
    ("sklearn",         "scikit-learn"),
    ("xgboost",         "xgboost"),
    ("lightgbm",        "lightgbm"),
    ("imblearn",        "imbalanced-learn"),
    ("optuna",          "optuna"),
    ("shap",            "shap"),
    ("matplotlib",      "matplotlib"),
    ("seaborn",         "seaborn"),
    ("fastapi",         "fastapi"),
    ("uvicorn",         "uvicorn"),
    ("joblib",          "joblib"),
]

REQUIRED_FOLDERS = [
    "data", "notebooks", "src", "models", "outputs", "images", "serving", "apps"
]

print("=" * 55)
print("  Credit Card Fraud Detection — Setup Verifier")
print("=" * 55)

# ── Python version ────────────────────────────────────────────────────────────
version = sys.version_info
status  = "✅" if version >= (3, 10) else "❌"
print(f"\n{status} Python {version.major}.{version.minor}.{version.micro}", end="")
if version < (3, 10):
    print("  ← Please upgrade to 3.10+")
else:
    print()

# ── Packages ──────────────────────────────────────────────────────────────────
print("\n📦 Packages:")
missing = []
for import_name, pip_name in REQUIRED_PACKAGES:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"   ✅ {pip_name:<22} {ver}")
    except ImportError:
        print(f"   ❌ {pip_name:<22} NOT INSTALLED")
        missing.append(pip_name)

# ── Folders ───────────────────────────────────────────────────────────────────
print("\n📁 Folders:")
for folder in REQUIRED_FOLDERS:
    exists = Path(folder).is_dir()
    icon   = "✅" if exists else "❌"
    print(f"   {icon} {folder}/")

# ── Data files ────────────────────────────────────────────────────────────────
print("\n📄 Data files:")
for f in ["data/transactions.csv", "data/transactions.parquet"]:
    p = Path(f)
    if p.exists():
        size_kb = p.stat().st_size // 1024
        print(f"   ✅ {f}  ({size_kb:,} KB)")
    else:
        print(f"   ⚠️  {f}  — run: python generate_data.py")

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if missing:
    print("❌ Missing packages — run:")
    print(f"   pip install {' '.join(missing)}")
else:
    print("✅ All checks passed! You're ready to build.")
    print("\n   Next step:")
    print("   python generate_data.py")
print("=" * 55)"""
verify_setup.py
───────────────
Run this FIRST to confirm your environment is ready.

    python verify_setup.py

Checks:
    ✅ Python version >= 3.10
    ✅ All required packages installed
    ✅ Folder structure exists
    ✅ Data files (after generate_data.py)
"""

import sys
import importlib
from pathlib import Path

REQUIRED_PACKAGES = [
    ("pandas",          "pandas"),
    ("numpy",           "numpy"),
    ("sklearn",         "scikit-learn"),
    ("xgboost",         "xgboost"),
    ("lightgbm",        "lightgbm"),
    ("imblearn",        "imbalanced-learn"),
    ("optuna",          "optuna"),
    ("shap",            "shap"),
    ("matplotlib",      "matplotlib"),
    ("seaborn",         "seaborn"),
    ("fastapi",         "fastapi"),
    ("uvicorn",         "uvicorn"),
    ("joblib",          "joblib"),
]

REQUIRED_FOLDERS = [
    "data", "notebooks", "src", "models", "outputs", "images", "serving", "apps"
]

print("=" * 55)
print("  Credit Card Fraud Detection — Setup Verifier")
print("=" * 55)

# ── Python version ────────────────────────────────────────────────────────────
version = sys.version_info
status  = "✅" if version >= (3, 10) else "❌"
print(f"\n{status} Python {version.major}.{version.minor}.{version.micro}", end="")
if version < (3, 10):
    print("  ← Please upgrade to 3.10+")
else:
    print()

# ── Packages ──────────────────────────────────────────────────────────────────
print("\n📦 Packages:")
missing = []
for import_name, pip_name in REQUIRED_PACKAGES:
    try:
        mod = importlib.import_module(import_name)
        ver = getattr(mod, "__version__", "?")
        print(f"   ✅ {pip_name:<22} {ver}")
    except ImportError:
        print(f"   ❌ {pip_name:<22} NOT INSTALLED")
        missing.append(pip_name)

# ── Folders ───────────────────────────────────────────────────────────────────
print("\n📁 Folders:")
for folder in REQUIRED_FOLDERS:
    exists = Path(folder).is_dir()
    icon   = "✅" if exists else "❌"
    print(f"   {icon} {folder}/")

# ── Data files ────────────────────────────────────────────────────────────────
print("\n📄 Data files:")
for f in ["data/transactions.csv", "data/transactions.parquet"]:
    p = Path(f)
    if p.exists():
        size_kb = p.stat().st_size // 1024
        print(f"   ✅ {f}  ({size_kb:,} KB)")
    else:
        print(f"   ⚠️  {f}  — run: python generate_data.py")

# ── Final verdict ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
if missing:
    print("❌ Missing packages — run:")
    print(f"   pip install {' '.join(missing)}")
else:
    print("✅ All checks passed! You're ready to build.")
    print("\n   Next step:")
    print("   python generate_data.py")
print("=" * 55)