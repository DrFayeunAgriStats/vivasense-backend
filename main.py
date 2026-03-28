"""
Entry point for Render's `uvicorn main:app` start command.
Changes working directory to genetics-module/ so that relative
R script paths (e.g. source("vivasense_genetics.R")) resolve correctly.
"""
import os
import sys

_module_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "genetics-module")
os.chdir(_module_dir)
sys.path.insert(0, _module_dir)

from app_genetics import app  # noqa: F401  – re-exported for uvicorn
