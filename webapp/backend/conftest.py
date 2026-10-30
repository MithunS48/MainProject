"""
conftest.py
-----------
Adds the backend root to sys.path so that `from app.xxx import yyy`
works correctly when running pytest from webapp/backend/.
"""
import sys
from pathlib import Path

# Ensure 'app' package is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))
