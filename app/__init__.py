"""spatial_registration GUI (task 14): PyQt5, standalone, UI code only.

All registration, session and export logic stays in src/ and is called
through pipeline.run_sample / run_batch and session.save_session /
load_session / export_result. Run with:  python -m app  (from
spatial_registration/).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
