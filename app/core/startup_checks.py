"""
app/core/startup_checks.py

Verifies that Rscript is accessible and that every required R package is
loadable.  Call run_startup_checks() once during the FastAPI startup event.

Raises RuntimeError with a descriptive message if any check fails, which
causes Render (and any other PaaS) to report a failed deployment rather than
silently serving a half-broken service.
"""

import logging
import shutil
import subprocess
from typing import List

logger = logging.getLogger(__name__)

REQUIRED_R_PACKAGES: List[str] = [
    "car",
    "lme4",
    "emmeans",
    "multcomp",
    "lmerTest",
    "pbkrtest",
    "agricolae",
    "sommer",
    "dplyr",
    "tidyr",
    "ggplot2",
    "jsonlite",
    "readr",
    "stringr",
    "purrr",
    "broom",
    "rlang",
    "tibble",
]

# Inline R snippet – checks each package and prints "MISSING: pkg1,pkg2" if any
# are absent, then exits non-zero.
_CHECK_TEMPLATE = """\
pkgs <- c({pkgs})
missing <- pkgs[!sapply(pkgs, requireNamespace, quietly = TRUE)]
if (length(missing) > 0) {{
  cat("MISSING:", paste(missing, collapse=","), "\\n")
  quit(status = 1)
}} else {{
  cat("OK\\n")
}}
"""


def _build_r_check_script() -> str:
    quoted = ", ".join(f'"{p}"' for p in REQUIRED_R_PACKAGES)
    return _CHECK_TEMPLATE.format(pkgs=quoted)


def run_startup_checks() -> None:
    """
    Check Rscript availability and verify all required R packages are installed.

    Raises:
        RuntimeError: if Rscript is not found or any package fails to load.
    """
    # ── 1. Rscript must be on PATH ────────────────────────────────────────────
    rscript = shutil.which("Rscript")
    if not rscript:
        raise RuntimeError(
            "Rscript not found in PATH. "
            "Ensure r-base is installed in the Docker image and the PATH is correct."
        )
    logger.info("startup_checks: Rscript found at %s", rscript)

    # ── 2. Verify every required R package is loadable ────────────────────────
    script = _build_r_check_script()
    try:
        result = subprocess.run(
            [rscript, "--vanilla", "-e", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("R package verification timed out after 60 s.") from exc

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0 or stdout.startswith("MISSING:"):
        detail = stdout or stderr or f"exit code {result.returncode}"
        raise RuntimeError(
            f"R package verification failed — {detail}. "
            "Rebuild the Docker image to ensure all packages are installed."
        )

    logger.info(
        "startup_checks: all %d required R packages verified. (%s)",
        len(REQUIRED_R_PACKAGES),
        stdout,
    )
