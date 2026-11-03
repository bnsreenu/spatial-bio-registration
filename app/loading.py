"""Reading input summaries and thumbnails off the UI thread, with a progress dialog (task 19).

The work runs in a plain Python thread; the window stays responsive because a local event loop keeps running
while it works. Results go into summary.py's caches, so the cards that need them afterwards are instant.
Cancel stops after the file being read; anything not read yet is read when its card is shown."""
from __future__ import annotations

import threading
from typing import Dict, List, Tuple

from PyQt5.QtCore import QEventLoop, Qt, QTimer
from PyQt5.QtWidgets import QProgressDialog

from app import summary

POLL_MS = 40


def warm_cache(items: List[Tuple[str, List[Dict]]], progress: Dict) -> None:
    """Worker body: describe each layer and build its thumbnail. items = [(sample id, layers)]."""
    for i, (_sid, layers) in enumerate(items):
        progress["index"] = i
        for layer in layers:
            if progress["cancel"]:
                progress["done"] = True
                return
            desc, _err = summary.describe(layer)
            if desc is not None:
                summary.thumbnail(desc, layer)
        progress["completed"] = i + 1
    progress["done"] = True


def read_with_progress(win, items: List[Tuple[str, List[Dict]]], title: str, noun: str = "sample",
                       show_after_ms: int = 0) -> bool:
    """Read all items with a progress dialog ("Reading sample 3 of 40...") and a Cancel button.
    Returns True if everything was read, False if the user cancelled."""
    total = len(items)
    if total == 0:
        return True
    dialog = QProgressDialog(f"Reading {noun} 1 of {total}...", "Cancel", 0, total, win)
    dialog.setWindowTitle(title)
    dialog.setWindowModality(Qt.WindowModal)
    dialog.setMinimumDuration(show_after_ms)
    dialog.setAutoClose(False)
    dialog.setAutoReset(False)
    dialog.setMinimumWidth(380)
    dialog.setValue(0)
    if show_after_ms == 0:
        dialog.show()
    progress = {"index": 0, "completed": 0, "done": False, "cancel": False, "error": None}

    def work():
        try:
            warm_cache(items, progress)
        except Exception as exc:  # noqa: BLE001 - the cards read lazily and report their own problems
            progress["error"] = exc
            progress["done"] = True

    thread = threading.Thread(target=work, daemon=True)
    loop = QEventLoop()

    def tick():
        if dialog.wasCanceled():
            progress["cancel"] = True
        dialog.setLabelText(f"Reading {noun} {min(progress['index'] + 1, total)} of {total}...")
        dialog.setValue(progress["completed"])
        if progress["done"]:
            loop.quit()

    timer = QTimer()
    timer.setInterval(POLL_MS)
    timer.timeout.connect(tick)
    thread.start()
    timer.start()
    loop.exec_()
    timer.stop()
    thread.join()
    cancelled = progress["cancel"]
    dialog.close()
    dialog.deleteLater()
    return not cancelled
