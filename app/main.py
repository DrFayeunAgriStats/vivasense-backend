"""
Entry point when Render runs `uvicorn app.main:app`.
Delegates entirely to genetics-module/app_genetics.py — do not add routes here.
"""
import os
import sys

_genetics_dir = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "genetics-module")
)
os.chdir(_genetics_dir)
sys.path.insert(0, _genetics_dir)

from app_genetics import app  # noqa: F401  – re-exported for uvicorn
