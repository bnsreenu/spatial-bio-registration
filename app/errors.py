"""Error logging for the GUI: the user sees a plain message, the exception
text goes to a log file (and to a Details section). No Qt imports."""
from __future__ import annotations

import time
import traceback
from pathlib import Path
from typing import Optional

LOG_NAME = "gui_errors.log"
_log_folder: Optional[Path] = None


def set_output_folder(folder) -> None:
    """Errors are logged in <output folder>/logs once an output folder is set."""
    global _log_folder
    _log_folder = Path(folder) / "logs" if folder else None


def log_path() -> Path:
    """<output folder>/logs/gui_errors.log, else gui_errors.log in the app config folder (same base as config.py)."""
    if _log_folder is not None:
        return _log_folder / LOG_NAME
    from app import config
    return config.config_path().parent / LOG_NAME


OVERLAY_FAILED = (
    "The overlay could not be built for this layer. Check that the linked images still exist and open "
    "correctly on the Samples screen. Technical detail is in the Details section."
)
INPUT_UNREADABLE = "This input could not be opened or recognized."
RUN_FAILED = "Registration failed for this sample."
LOW_ADVICE = (
    "What to do: check that the right images are linked on the Samples screen, then use landmarks "
    "(coming next)."
)


def exception_text(exc: BaseException) -> str:
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)).strip()


def log_text(context: str, text: str) -> None:
    """Append text to the log file; never raises (logging must not break the UI)."""
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {context}\n{text}\n\n")
    except OSError:
        pass


def log_exception(context: str, exc: BaseException) -> str:
    """Log exc and return its full text (for a Details section)."""
    text = exception_text(exc)
    log_text(context, text)
    return text
