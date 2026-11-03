"""Main window and the controller functions that connect the screens to
src/. MainWindow is the only class here (Qt requires the subclass); every
other piece of behavior is a function taking the window."""
from __future__ import annotations

import copy
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from PyQt5.QtCore import QEventLoop, Qt, QTimer
from PyQt5.QtGui import QColor, QPixmap
from PyQt5.QtWidgets import QAction, QApplication, QListWidgetItem, QMainWindow

import pipeline
import reference as reference_mod
import review_overlay as ro
import session
import version
from app import (
    config, dialogs, errors, help_text, loading, review_status as st_review, state as st, summary, theme,
)
from app.screens import (
    build_data_screen, build_help_dialog, build_review_screen, build_shell, dot_icon, status_pixmap,
)
from app.widgets import BatchThread, array_to_qimage

PAGE_DATA, PAGE_REVIEW = 0, 1
SHOW_AFTER_MS = 400  # single-file reads show the progress dialog only if they take longer than this
FLICKER_MS = 600
BADGE_COLORS = {"high": "#2e8b57", "medium": "#d98e04", "low": "#c0392b", "failed": "#555555", "cancelled": "#555555"}
DOT_COLORS = {"incomplete": theme.GRAY_DOT, "ready": theme.AMBER, "running": theme.LIGHT_TEAL,
              "high": theme.GREEN, "medium": theme.AMBER, "low": theme.RED}
THUMB_PX = 96


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spatial registration")
        self.resize(1280, 820)
        self.setAcceptDrops(True)
        self.state = st.new_state()
        init_output_folder(self)
        self.thread: Optional[BatchThread] = None
        self.run_ids: List[str] = []
        self.done_count = 0
        self.step_i = 0
        self.run_layers = {}
        self.run_message = ""
        self.save_problem = ""
        self.overlay_cache = {}
        self.overlay_errors = {}
        self.layer_keys = []
        self.review_stale = False
        self.flicker_on = False
        self.flicker_timer = QTimer(self)
        self.flicker_timer.setInterval(FLICKER_MS)
        self.flicker_timer.timeout.connect(lambda: flicker_tick(self))
        self.shell = build_shell()
        self.s1 = build_data_screen()
        self.s3 = build_review_screen()
        self.shell.pages.addWidget(self.s1.widget)
        self.shell.pages.addWidget(self.s3.widget)
        self.setCentralWidget(self.shell.widget)
        build_menu(self)
        connect_signals(self)
        refresh_all(self)
        refresh_review_list(self)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            handle_drop(self, paths, None)
            event.acceptProposedAction()

    def closeEvent(self, event):
        self.flicker_timer.stop()
        if self.thread is not None and self.thread.isRunning():
            self.thread.cancel_event.set()
            self.thread.wait(60000)
        super().closeEvent(event)


# ---------------------------------------------------------------- setup

def build_menu(win) -> None:
    menu = win.menuBar().addMenu("&File")
    for text, fn in (("&New", new_session), ("&Open session...", open_session), ("&Save session...", save_current_session)):
        action = QAction(text, win)
        action.triggered.connect(lambda _checked=False, f=fn: f(win))
        menu.addAction(action)
        setattr(win, "action_" + text.strip("&.").split()[0].lower(), action)
    help_menu = win.menuBar().addMenu("&Help")
    for attr, text, fn in (("action_guide", "&Quick guide", show_guide), ("action_about", "&About", show_about)):
        action = QAction(text, win)
        action.triggered.connect(lambda _checked=False, f=fn: f(win))
        help_menu.addAction(action)
        setattr(win, attr, action)


def show_guide(win) -> None:
    dlg, _topics, _text = build_help_dialog(win)
    dialogs.show_dialog(dlg)


def show_about(win) -> None:
    dialogs.info(win, "About", help_text.ABOUT_TEXT)


def connect_pick(win, card, modality: str) -> None:
    """Wire a card's Browse and Replace controls to the pickers."""
    if modality == "xenium":
        card.browse.clicked.connect(lambda: pick_for_card(win, modality, "folder"))
        card.replace.clicked.connect(lambda: pick_for_card(win, modality, "folder"))
        return
    card.browse_file.triggered.connect(lambda: pick_for_card(win, modality, "file"))
    card.browse_zarr.triggered.connect(lambda: pick_for_card(win, modality, "folder"))
    card.replace_file.triggered.connect(lambda: pick_for_card(win, modality, "file"))
    card.replace_zarr.triggered.connect(lambda: pick_for_card(win, modality, "folder"))


def connect_signals(win) -> None:
    s1, s3 = win.s1, win.s3
    for i, b in enumerate(win.shell.steps):
        b.clicked.connect(lambda _c=False, idx=i: show_page(win, idx))
    s1.empty_folder.clicked.connect(lambda: add_folder(win))
    s1.empty_add.clicked.connect(lambda: add_sample(win))
    s1.drop.paths_dropped.connect(lambda paths: handle_drop(win, paths, None))
    s1.add_sample.clicked.connect(lambda: add_sample(win))
    s1.add_folder.clicked.connect(lambda: add_folder(win))
    s1.remove_sample.clicked.connect(lambda: remove_sample(win))
    s1.out_change.clicked.connect(lambda: change_output_folder(win))
    s1.samples.currentRowChanged.connect(lambda _r: show_sample(win))
    s1.sample_name.editingFinished.connect(lambda: rename_sample(win))
    for modality, card in s1.cards.items():
        connect_pick(win, card, modality)
        card.frame.paths_dropped.connect(lambda paths, m=modality: handle_drop(win, paths, m))
        card.remove.clicked.connect(lambda _c=False, m=modality: clear_card(win, m))
        card.px_set.clicked.connect(lambda _c=False, m=modality: apply_pixel_size(win, m))
    for modality, b in s1.segments.items():
        b.clicked.connect(lambda _c=False, m=modality: choose_reference(win, m))
    s1.adv_toggle.toggled.connect(lambda on: toggle_advanced(win, on))
    s1.target.valueChanged.connect(lambda v: win.state["settings"].__setitem__("target_um", float(v)))
    s1.run.clicked.connect(lambda: start_run(win, all_ready=False))
    s1.run_all.clicked.connect(lambda: start_run(win, all_ready=True))
    s1.cancel.clicked.connect(lambda: cancel_run(win))
    s1.open_review.clicked.connect(lambda: show_page(win, PAGE_REVIEW))
    s3.samples.currentRowChanged.connect(lambda _r: show_review(win))
    s3.layer.currentIndexChanged.connect(lambda _i: on_layer_change(win))
    for b in s3.mode_buttons.values():
        b.toggled.connect(lambda _on: on_mode_change(win))
    s3.slider.valueChanged.connect(lambda _v: render_view(win))
    for b in s3.color_buttons.values():
        b.toggled.connect(lambda _on: render_view(win))
    s3.markers.toggled.connect(lambda on: set_markers_visible(win, on))
    s3.change_ref.clicked.connect(lambda: show_page(win, PAGE_DATA))
    link_panes(win)
    s3.export_one.clicked.connect(lambda: export_flow(win, [review_sample_id(win)]))
    s3.export_accepted.clicked.connect(lambda: export_flow(win, None))
    s3.accept.clicked.connect(lambda: set_review(win, review_sample_id(win), st_review.ACCEPTED))
    s3.reject.clicked.connect(lambda: set_review(win, review_sample_id(win), st_review.REJECTED))
    s3.accept_high.clicked.connect(lambda: accept_all_high(win))


# ---------------------------------------------------------------- shell

def show_page(win, index: int) -> None:
    """Switch step. Review is available only once a sample has a result."""
    if index == PAGE_REVIEW and not win.state["results"]:
        return
    win.shell.pages.setCurrentIndex(index)
    if index == PAGE_REVIEW and win.review_stale:
        show_review(win)
    update_steps(win)


def update_steps(win) -> None:
    current = win.shell.pages.currentIndex()
    win.shell.steps[PAGE_REVIEW].setEnabled(bool(win.state["results"]))
    for i, b in enumerate(win.shell.steps):
        theme.set_prop(b, "active", i == current)


def is_running(win) -> bool:
    return win.thread is not None and win.thread.isRunning()


# ---------------------------------------------------------------- Data screen: samples

def current_sample(win) -> Optional[dict]:
    item = win.s1.samples.currentItem()
    return None if item is None else st.find_sample(win.state, item.data(Qt.UserRole))


def sample_status(win, sample: dict):
    """(level, plain text) for the sidebar: gray incomplete, amber ready,
    after a run green/amber/red by confidence."""
    sid = sample["sample_id"]
    if is_running(win) and sid in win.run_ids:
        index = win.run_ids.index(sid)
        return ("running", "Running") if index == win.done_count else ("ready", "Waiting to run") if index > win.done_count else ("ready", "Finishing")
    reasons = st.blocked_reasons(sample)
    if reasons:
        return "incomplete", "Incomplete"
    res = win.state["results"].get(sid)
    if res is not None and win.state["inputs"].get(sid) == sample["layers"]:
        if res["status"] == "ok":
            return res["confidence"], f"Registered, confidence {res['confidence']}"
        if res["status"] == "cancelled":
            return "ready", "Cancelled, ready to run again"
        return "low", "Registration failed"
    return "ready", "Ready" if res is None else "Ready (inputs changed since the last run)"


def update_sidebar_status(win) -> None:
    lst = win.s1.samples
    for i in range(lst.count()):
        item = lst.item(i)
        sample = st.find_sample(win.state, item.data(Qt.UserRole))
        if sample is None:
            continue
        level, text = sample_status(win, sample)
        item.setText(f"{sample['sample_id']}\n{text}")
        item.setIcon(dot_icon(DOT_COLORS[level]))
        item.setToolTip("; ".join(st.blocked_reasons(sample)) or text)


def refresh_sidebar(win, select: Optional[str] = None) -> None:
    lst = win.s1.samples
    keep = select or (current_sample(win) or {}).get("sample_id")
    lst.blockSignals(True)
    lst.clear()
    for sample in win.state["samples"]:
        item = QListWidgetItem()
        item.setData(Qt.UserRole, sample["sample_id"])
        lst.addItem(item)
    lst.blockSignals(False)
    update_sidebar_status(win)
    win.s1.stack.setCurrentIndex(1 if win.state["samples"] else 0)
    ids = [s["sample_id"] for s in win.state["samples"]]
    if ids:
        lst.blockSignals(True)
        lst.setCurrentRow(ids.index(keep) if keep in ids else 0)
        lst.blockSignals(False)
    show_sample(win)


def refresh_all(win) -> None:
    refresh_output_folder(win)
    refresh_sidebar(win)
    update_steps(win)


def add_sample(win) -> None:
    sample = st.make_sample(st.unique_id(win.state, "sample"))
    win.state["samples"].append(sample)
    refresh_sidebar(win, select=sample["sample_id"])


def add_folder(win) -> None:
    folder = dialogs.choose_folder(win, "Folder of samples")
    if not folder:
        return
    lines, count = st.folder_proposal(folder)
    if not count:
        dialogs.info(win, "No samples found",
                     "No sample folders were found in that folder. Choose the folder that contains one folder per "
                     "sample, each with he, comet and xenium subfolders. See Help > Quick guide, \"How to organize "
                     "files\".")
        return
    text = (f"Found {summary.plural(count, 'sample')} in {Path(folder).name}:\n\n" + "\n".join(lines)
            + f"\n\nAdd {'it' if count == 1 else 'them'}? You can check and change every card afterwards.")
    if not dialogs.confirm(win, "Add folder of samples", text):
        return
    ids = st.propose_from_folder(win.state, folder)
    summary.clear_cache()
    win.overlay_cache.clear()
    items = [(s["sample_id"], [l for l in s["layers"].values()]) for s in win.state["samples"] if s["sample_id"] in ids]
    loading.read_with_progress(win, items, "Reading samples")
    refresh_sidebar(win, select=ids[0] if ids else None)


def remove_sample(win) -> None:
    sample = current_sample(win)
    if sample is None or is_running(win):
        return
    sid = sample["sample_id"]
    if sid in win.state["results"] and not dialogs.confirm(
            win, "Remove sample", f"{sid} has a registration result. Remove it and its result?", default_yes=False):
        return
    win.state["samples"].remove(sample)
    for key in ("results", "inputs", "review", "exports"):
        win.state[key].pop(sid, None)
    win.overlay_cache.clear()
    refresh_sidebar(win)
    refresh_review_list(win)
    update_steps(win)


def rename_sample(win) -> None:
    sample = current_sample(win)
    if sample is None:
        return
    new = win.s1.sample_name.text().strip()
    old = sample["sample_id"]
    if not new or new == old:
        win.s1.sample_name.setText(old)
        return
    if st.find_sample(win.state, new) is not None:
        dialogs.warn(win, "Sample name", f"A sample named {new} already exists.")
        win.s1.sample_name.setText(old)
        return
    sample["sample_id"] = new
    for key in ("results", "inputs", "review", "exports"):
        if old in win.state[key]:
            win.state[key] = {(new if k == old else k): v for k, v in win.state[key].items()}
    if new in win.state["results"]:
        win.state["results"][new]["sample_id"] = new
    win.overlay_cache.clear()
    refresh_sidebar(win, select=new)
    refresh_review_list(win)


# ---------------------------------------------------------------- Data screen: cards

def show_sample(win) -> None:
    s1 = win.s1
    win.run_message = ""
    sample = current_sample(win)
    s1.sample_name.blockSignals(True)
    s1.sample_name.setText("" if sample is None else sample["sample_id"])
    s1.sample_name.blockSignals(False)
    s1.sample_name.setEnabled(sample is not None)
    s1.remove_sample.setEnabled(sample is not None)
    s1.proposed.setText("Proposed from the folder: please check each card." if sample and sample["proposed"] else "")
    for modality, card in s1.cards.items():
        show_card(win, sample, modality, card)
    refresh_reference(win)
    refresh_footer(win)


def clear_thumb(card, text: str = "") -> None:
    card.thumb.setPixmap(QPixmap())
    card.thumb.setText(text)


def show_candidates(win, sample: Optional[dict], modality: str, card) -> None:
    """On an empty card: when the folder held several candidates for this
    modality, say so and offer them in a menu (nothing is chosen for the user)."""
    found = [] if sample is None else sample.get("candidates", {}).get(modality, [])
    card.cand_menu.clear()
    card.cand_label.setVisible(bool(found))
    card.cand_button.setVisible(bool(found))
    if not found:
        return
    card.cand_label.setText(st.candidates_text(modality, len(found)))
    for path in found:
        action = card.cand_menu.addAction(Path(path).name)
        action.setToolTip(path)
        action.triggered.connect(lambda _c=False, p=path: choose_candidate(win, sample, modality, p))


def choose_candidate(win, sample: dict, modality: str, path: str) -> None:
    assign_layer(win, sample, modality, path, proposed_note="chosen from several candidates in the folder")
    refresh_sidebar(win, select=sample["sample_id"])


def show_card(win, sample: Optional[dict], modality: str, card) -> None:
    layer = None if sample is None else sample["layers"].get(modality)
    card.px_row.setVisible(False)
    show_candidates(win, sample, modality, card)
    if layer is None:
        card.stack.setCurrentIndex(0)
        theme.set_prop(card.frame, "state", "empty")
        return
    card.stack.setCurrentIndex(1)
    desc, err = summary.describe(layer)
    proposed = modality in sample["proposed"]
    card.name.setText(Path(layer["path"]).name + ("  (proposed)" if proposed else ""))
    card.name.setToolTip(layer["path"] + ("\n" + sample["proposed"][modality] if proposed else ""))
    blocking, warnings = summary.card_messages(desc, err, layer)
    info: List[str] = []
    if modality == "xenium":
        lines = summary.xenium_files(layer["path"])
        card.facts.setText("Xenium output folder")
        card.files.setText("Found: " + "; ".join(lines) if lines else "No known Xenium files found in this folder.")
        card.files.setVisible(True)
        if summary.xenium_notice(desc):
            info.append(summary.XENIUM_NO_IMAGE_CARD)
        clear_thumb(card, "Xenium")
    else:
        card.facts.setText(summary.facts_line(desc) if desc else "")
        card.files.setVisible(False)
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            arr = summary.thumbnail(desc, layer)
        finally:
            QApplication.restoreOverrideCursor()
        if arr is None:
            clear_thumb(card, "No preview")
        else:
            card.thumb.setText("")
            card.thumb.setPixmap(QPixmap.fromImage(array_to_qimage(arr)).scaled(
                THUMB_PX, THUMB_PX, Qt.KeepAspectRatio, Qt.SmoothTransformation))
    scale_note = None if modality == "xenium" else summary.level_scale_note(desc)
    card.note.setText(scale_note or "")
    card.note_row.setVisible(bool(scale_note))
    kind = "block" if blocking else "warn" if warnings else "ok"
    card.icon.setPixmap(status_pixmap(kind))
    messages = blocking + warnings + info
    card.msg.setText("\n".join(messages))
    card.msg.setVisible(bool(messages))
    theme.set_prop(card.msg, "role", "block" if blocking else "warn" if warnings else "muted")
    theme.set_prop(card.frame, "state", "block" if blocking else "warn" if (warnings or proposed) else "loaded")
    if desc is not None and modality != "xenium" and summary.pixel_size_missing(desc):
        card.px_row.setVisible(True)
        card.px_spin.setValue(0.0)
    card.details.text.setPlainText(summary.level_table(desc) if desc else summary.details_text(None, layer))


def assign_layer(win, sample: dict, modality: str, path: str, proposed_note: Optional[str] = None) -> None:
    st.set_layer(sample, modality, path, proposed_note=proposed_note)
    summary.clear_cache()
    win.overlay_cache.clear()
    loading.read_with_progress(win, [(sample["sample_id"], [sample["layers"][modality]])], "Reading file",
                               noun="file", show_after_ms=SHOW_AFTER_MS)


def pick_for_card(win, modality: str, kind: str) -> None:
    sample = current_sample(win)
    if sample is None:
        return
    title = "Choose the " + ("Xenium output folder" if modality == "xenium" else summary.name(modality) + (" zarr folder" if kind == "folder" else " file"))
    chooser = dialogs.choose_folder if kind == "folder" else dialogs.choose_file
    path = chooser(win, title)
    if path:
        assign_layer(win, sample, modality, path)
        refresh_sidebar(win)


def handle_drop(win, paths: List[str], modality: Optional[str]) -> None:
    """Files or folders dropped on a card (modality given) or anywhere else
    (modality None: the column is guessed from the content)."""
    if is_running(win):
        return
    rest = list(paths)
    if modality is None:
        loading.read_with_progress(win, [(Path(p).name, [{"path": p, "pixel_size_um": None}]) for p in rest],
                                   "Reading files", noun="file", show_after_ms=SHOW_AFTER_MS)
    sample = current_sample(win)
    select = None
    if modality is not None:
        if sample is None:
            sample = st.make_sample(st.unique_id(win.state, "sample"))
            win.state["samples"].append(sample)
        assign_layer(win, sample, modality, rest.pop(0))
        select = sample["sample_id"]
    for path in rest:
        select = add_dropped_path(win, path, sample) or select
    refresh_sidebar(win, select=select)


def add_dropped_path(win, path: str, target: Optional[dict]) -> Optional[str]:
    """Add one dropped path; returns the id of the sample it went to."""
    desc, _err = summary.describe({"path": path, "pixel_size_um": None})
    if desc is None and Path(path).is_dir():
        ids = st.propose_from_folder(win.state, path)
        if ids:
            return ids[0]
        dialogs.info(win, "Nothing recognized", f"No H&E, COMET or Xenium data was recognized in {Path(path).name}.")
        return None
    modality = st.guess_modality(path)
    if target is None or modality in target["layers"]:
        target = st.make_sample(st.unique_id(win.state, Path(path).stem or "sample"))
        win.state["samples"].append(target)
    assign_layer(win, target, modality, path, proposed_note="column guessed from file content")
    return target["sample_id"]


def clear_card(win, modality: str) -> None:
    sample = current_sample(win)
    if sample is not None:
        st.clear_layer(sample, modality)
        win.overlay_cache.clear()
        refresh_sidebar(win)


def apply_pixel_size(win, modality: str) -> None:
    sample = current_sample(win)
    card = win.s1.cards[modality]
    value = card.px_spin.value()
    if sample is None or modality not in sample["layers"]:
        return
    if value <= 0:
        dialogs.warn(win, "Pixel size", "Enter a pixel size greater than zero.")
        return
    sample["layers"][modality]["pixel_size_um"] = float(value)
    summary.clear_cache()
    win.overlay_cache.clear()
    refresh_sidebar(win)


# ---------------------------------------------------------------- Data screen: reference

def refresh_reference(win) -> None:
    s1 = win.s1
    sample = current_sample(win)
    loaded = st.loaded_references(sample) if sample else []
    chosen, reason = st.chosen_reference(sample) if sample else (None, "")
    s1.seg_group.setExclusive(False)
    for modality, b in s1.segments.items():
        b.setEnabled(modality in loaded and not is_running(win))
        b.blockSignals(True)
        b.setChecked(modality == chosen)
        b.blockSignals(False)
    s1.seg_group.setExclusive(True)
    s1.ref_reason.setText(reason or "Add an image to choose the reference.")


def choose_reference(win, modality: str) -> None:
    sample = current_sample(win)
    if sample is None or modality not in sample["layers"]:
        return
    sample["reference"] = modality
    autosave_session(win, sample["sample_id"])
    refresh_reference(win)
    win.review_stale = True


def toggle_advanced(win, on: bool) -> None:
    win.s1.adv_panel.setVisible(on)
    win.s1.adv_toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)


# ---------------------------------------------------------------- Data screen: run footer

def set_role(widget, role: str) -> None:
    """Switch a button between the primary and secondary look."""
    if widget.property("role") != role:
        widget.setProperty("role", role)
        widget.style().unpolish(widget)
        widget.style().polish(widget)


def refresh_footer(win) -> None:
    s1 = win.s1
    running = is_running(win)
    sample = current_sample(win)
    n_ready = len(st.runnable_samples(win.state))
    if not running:
        if win.run_message:
            text = win.run_message
        elif sample is None:
            text = "Add a sample to begin."
        else:
            reasons = st.blocked_reasons(sample)
            text = ("Not ready: " + "; ".join(reasons) + ".") if reasons else f"Ready: {st.linked_text(sample)} linked"
        s1.status.setText(text)
    ready_here = sample is not None and not st.blocked_reasons(sample)
    s1.run.setEnabled(ready_here and not running)
    s1.run_all.setEnabled(n_ready > 0 and not running)
    s1.run_all.setText("Auto-register all ready samples" + (f" ({n_ready})" if n_ready > 1 else ""))
    s1.cancel.setVisible(running)
    has_results = bool(win.state["results"])
    s1.open_review.setVisible(has_results and not running)
    for b in (s1.run, s1.run_all):
        set_role(b, "secondary" if has_results else "primary" if b is s1.run else "secondary")
    s1.bar.setVisible(running)
    s1.step.setVisible(running)


def existing_status_text(win, sid: str) -> Optional[str]:
    """The review status of a sample that already has a result or a session file (None if it has neither)."""
    if sid in win.state["results"]:
        return review_label(win, sid)
    path = session_path(win, sid)
    if path.is_file():
        try:
            data = session.load_session(path)
            return st_review.label(data["result"], st_review.clean(data.get("review")))
        except Exception as exc:  # noqa: BLE001 - an unreadable old file is simply replaced after asking
            errors.log_exception(f"read session file of {sid}", exc)
            return "unknown"
    return None


def confirm_reruns(win, runnable: List[dict]) -> List[dict]:
    """Ask before replacing a result or session file. Samples the user skips are left exactly as they are."""
    existing = [s for s in runnable if existing_status_text(win, s["sample_id"]) is not None]
    kept: List[dict] = []
    remember = None
    for sample in runnable:
        sid = sample["sample_id"]
        status = existing_status_text(win, sid)
        if status is None:
            kept.append(sample)
            continue
        if remember is None:
            left = len(existing) - existing.index(sample)
            text = (f"Re-run {sid}? This replaces its current result and resets its review status "
                    f"(currently: {status}).")
            rerun, every = dialogs.rerun_dialog(win, text, offer_all=left > 1)
            if every:
                remember = rerun
        else:
            rerun = remember
        if rerun:
            kept.append(sample)
    return kept

def start_run(win, all_ready: bool) -> None:
    if is_running(win):
        return
    s1 = win.s1
    runnable = st.runnable_samples(win.state)
    if not all_ready:
        current = current_sample(win)
        runnable = [current] if current in runnable else []
    if not runnable:
        s1.status.setText("Nothing to run. Link at least two images to this sample and fix anything marked on its cards.")
        return
    if not ensure_project_dir(win):
        return
    runnable = confirm_reruns(win, runnable)
    if not runnable:
        s1.status.setText("Nothing was re-run. The existing results and review statuses are unchanged.")
        return
    win.run_ids =  [s["sample_id"] for s in runnable]
    win.done_count = 0
    win.step_i = 0
    win.run_message = ""
    win.save_problem = ""
    s1.open_review.setVisible(False)
    s1.step.setText("")
    s1.bar.setValue(0)
    s1.status.setText(f"Running {summary.plural(len(runnable), 'sample')}.")
    s1.cancel.setEnabled(True)
    win.run_layers = {s["sample_id"]: copy.deepcopy(s["layers"]) for s in runnable}
    win.thread = BatchThread([st.to_run_dict(s) for s in runnable], win.state["settings"]["target_um"], parent=win)
    win.thread.progress.connect(lambda f, m: on_progress(win, f, m))
    win.thread.sample_done.connect(lambda res: on_sample_done(win, res))
    win.thread.batch_done.connect(lambda results, table: on_batch_done(win, results))
    win.thread.batch_error.connect(lambda msg: on_batch_error(win, msg))
    win.thread.start()
    refresh_footer(win)
    refresh_reference(win)
    update_sidebar_status(win)


def on_progress(win, fraction: float, message: str) -> None:
    n = max(len(win.run_ids), 1)
    if message.startswith("completed "):
        win.done_count += 1
        win.step_i = 0
        win.s1.bar.setValue(int(100 * win.done_count / n))
        update_sidebar_status(win)
        return
    match = re.match(r"registering (\w+) to (\w+)", message)
    if match and win.done_count < len(win.run_ids):
        win.step_i += 1
        sid = win.run_ids[win.done_count]
        total = max(len(win.run_layers.get(sid, {})) - 1, 1)
        win.s1.step.setText(f"{sid}: Registering {summary.name(match.group(1))} to {summary.name(match.group(2))}, "
                            f"{win.step_i} of {total}")


def cancel_run(win) -> None:
    if is_running(win):
        win.thread.cancel_event.set()
        win.s1.cancel.setEnabled(False)
        win.s1.status.setText("Cancelling: the registration step in progress finishes first.")


STATUS_WORDS = {"ok": "Done", "failed": "Failed", "cancelled": "Cancelled"}


def default_project_dir() -> str:
    """registration_projects inside the user's Documents folder (not created until the user accepts)."""
    from PyQt5.QtCore import QStandardPaths
    docs = QStandardPaths.writableLocation(QStandardPaths.DocumentsLocation) or str(Path.home() / "Documents")
    return str(Path(docs) / "registration_projects")


def init_output_folder(win) -> None:
    """The output folder is the remembered one, else the default. Nothing is created here."""
    win.state["project_dir"] = config.remembered_output_folder() or default_project_dir()
    errors.set_output_folder(win.state["project_dir"])


def output_folder(win) -> Path:
    errors.set_output_folder(win.state["project_dir"])
    return Path(win.state["project_dir"])


def sessions_dir(win) -> Path:
    return output_folder(win) / "sessions"


def results_dir(win) -> Path:
    return output_folder(win) / "results"


def output_folder_text(win) -> str:
    # zero-width spaces after separators let a long path wrap inside the narrow sidebar
    return str(output_folder(win)).replace("\\", "\\\u200b").replace("/", "/\u200b")


def refresh_output_folder(win) -> None:
    win.s1.out_path.setText(output_folder_text(win))
    win.s1.out_path.setToolTip(str(output_folder(win)))


def ensure_project_dir(win) -> bool:
    """The run uses the output folder shown on the Data screen; there is no folder prompt at run time. The only
    question is whether to create the folder the first time, if it does not exist yet."""
    folder = output_folder(win)
    if folder.is_dir():
        config.remember_output_folder(str(folder))
        return True
    if not dialogs.confirm(win, "Create output folder",
                           f"Create the output folder?\n\n{folder}\n\nSessions are saved in its sessions folder and "
                           "saved results in its results folder. You can change the folder on the Data screen."):
        win.s1.status.setText("The output folder was not created, so nothing was run. Use Change... under "
                              "Output folder to pick another one.")
        return False
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.log_exception("create output folder", exc)
        dialogs.warn(win, "Output folder", "The output folder could not be created. Use Change... to choose "
                                           "another folder.")
        return False
    config.remember_output_folder(str(folder))
    return True


def change_output_folder(win) -> None:
    if is_running(win):
        return
    chosen = dialogs.choose_folder(win, "Choose the output folder", str(output_folder(win)))
    if not chosen:
        return
    try:
        Path(chosen).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        errors.log_exception("create output folder", exc)
        dialogs.warn(win, "Output folder", "That folder could not be used. Choose another one.")
        return
    win.state["project_dir"] = str(chosen)
    config.remember_output_folder(str(chosen))
    refresh_output_folder(win)


def session_path(win, sid: str) -> Path:
    return sessions_dir(win) / f"{sid}_session.json"


def export_entry(win, sid: str) -> Optional[dict]:
    return win.state["exports"].get(sid)


def save_session_file(win, path, sid: str) -> None:
    session.save_session(path, win.state["inputs"].get(sid, {}), pixel_sizes_for(win, sid), win.state["results"][sid],
                         reference=reference_for(win, sid), review=st_review.entry(win.state["review"], sid),
                         export=export_entry(win, sid))


def autosave_session(win, sid: str) -> bool:
    """Write the sample's session file (inputs, result, reference, review status, export time) to
    <output folder>/sessions. Returns False, with the problem logged and shown in the footer, if it could not be
    written."""
    if win.state["results"].get(sid) is None or not win.state["project_dir"]:
        return False
    try:
        save_session_file(win, session_path(win, sid), sid)
        return True
    except Exception as exc:  # noqa: BLE001 - plain message shown, exception logged
        errors.log_exception(f"autosave of {sid}", exc)
        win.save_problem = ("A session file could not be saved to the output folder. Check that the folder "
                            "can be written to. Technical detail is in the log file.")
        win.run_message = win.save_problem
        refresh_footer(win)
        return False


def store_result(win, res: dict) -> None:
    sid = res["sample_id"]
    win.state["results"][sid] = res
    win.state["inputs"][sid] = win.run_layers.get(sid, {})
    st_review.set_status(win.state["review"], sid, st_review.UNREVIEWED)
    win.state["exports"].pop(sid, None)
    if res.get("error"):
        errors.log_text(f"registration of {sid} ended with status {res['status']}", str(res["error"]))
    if res["status"] == "ok":
        autosave_session(win, sid)


def on_sample_done(win, res: dict) -> None:
    """Called as each sample finishes: keep the result in the session and autosave. Nothing is saved to results."""
    store_result(win, res)
    win.overlay_cache.clear()
    update_sidebar_status(win)


def on_batch_done(win, results: list) -> None:
    for res in results:
        if win.state["results"].get(res["sample_id"]) is not res:
            store_result(win, res)
    win.overlay_cache.clear()
    cancelled = any(r["status"] == "cancelled" for r in results)
    win.s1.bar.setValue(win.s1.bar.value() if cancelled else 100)
    win.s1.step.setText("")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    refresh_review_list(win)
    refresh_all(win)
    base = (f"Registration finished ({n_ok} of {summary.plural(len(results), 'sample')} registered). "
            "Results are held in the session. Review them, then use Save accepted results to write the CSV files.")
    if n_ok == 0:
        reason = "The run was cancelled" if cancelled else "Every sample failed to register"
        base = (f"{reason}, so there is nothing to review. See the sample status on the Data screen; "
                "details are in the log file.")
    elif cancelled:
        base += " The run was cancelled, so you stay on this screen."
    win.run_message = base
    if win.save_problem:
        win.run_message += " " + win.save_problem
    end_run_ui(win)
    if n_ok >= 1 and not cancelled:
        refresh_review_list(win, select=next(r["sample_id"] for r in results if r["status"] == "ok"))
        show_page(win, PAGE_REVIEW)


def end_run_ui(win) -> None:
    win.thread = None if win.thread is None or not win.thread.isRunning() else win.thread
    refresh_footer(win)
    refresh_reference(win)
    update_sidebar_status(win)


def on_batch_error(win, message: str) -> None:
    errors.log_text("run_batch raised", message)
    win.s1.step.setText("")
    end_run_ui(win)
    win.run_message = "The run stopped because of an unexpected problem. Details were written to the log file."
    refresh_footer(win)


# ---------------------------------------------------------------- Review screen
def review_sample_id(win) -> Optional[str]:
    item = win.s3.samples.currentItem()
    return None if item is None else item.data(Qt.UserRole)


def review_reference(win, sid: str) -> Optional[str]:
    """The reference chosen on the Data screen for this sample, if it is part
    of the result, else the default (Xenium, else COMET, else H&E)."""
    res = win.state["results"].get(sid)
    if res is None:
        return None
    mods = reference_mod.modalities_in(res)
    sample = st.find_sample(win.state, sid)
    chosen = st.chosen_reference(sample)[0] if sample else None
    return chosen if chosen in mods else reference_mod.default_reference(mods)[0]


def composed_result(win, sid: str) -> dict:
    """The result expressed against the Review reference (the engine result
    itself when they are the same)."""
    res = win.state["results"][sid]
    ref = review_reference(win, sid)
    if res["status"] != "ok" or not ref:
        return res
    return reference_mod.apply_reference(res, ref, pixel_sizes_for(win, sid))


def review_dot(res: dict) -> str:
    """Dot color: red when the sample needs attention (low confidence or failed), else by confidence."""
    if st_review.needs_attention(res):
        return "low"
    return res["confidence"] if res["status"] == "ok" else res["status"]


def review_label(win, sid: str) -> str:
    return st_review.label(win.state["results"][sid], st_review.entry(win.state["review"], sid))


def fill_review_item(win, item, sid: str) -> None:
    res = win.state["results"][sid]
    status = review_label(win, sid)
    item.setIcon(dot_icon(DOT_COLORS.get(review_dot(res), theme.GRAY_DOT)))
    item.setText(f"{sid}\n{status.capitalize()}")
    conf = f"confidence {res['confidence']}" if res["status"] == "ok" else STATUS_WORDS.get(res["status"], res["status"]).lower()
    item.setToolTip(f"{sid}: {conf}; {status}")


def refresh_review_items(win) -> None:
    """Update the dots and status lines in place (no rebuild of the view)."""
    lst = win.s3.samples
    for i in range(lst.count()):
        item = lst.item(i)
        sid = item.data(Qt.UserRole)
        if sid in win.state["results"]:
            fill_review_item(win, item, sid)
    refresh_status_row(win)


def refresh_status_row(win) -> None:
    s3 = win.s3
    sid = review_sample_id(win)
    have = sid in win.state["results"]
    s3.status.setText(f"Review status: {review_label(win, sid).capitalize()}" if have else "")
    s3.export_state.setText(export_label(win, sid) if have else "")
    s3.accept.setEnabled(have)
    s3.reject.setEnabled(have)
    s3.accept_high.setEnabled(bool(win.state["results"]))
    s3.export_accepted.setEnabled(bool(win.state["results"]))


def set_review(win, sid: str, status: str) -> None:
    if sid not in win.state["results"]:
        return
    st_review.set_status(win.state["review"], sid, status)
    autosave_session(win, sid)
    refresh_review_items(win)


def accept_all_high(win) -> None:
    ids = st_review.bulk_candidates(win.state["results"], win.state["review"])
    if not ids:
        dialogs.info(win, "Accept all high-confidence",
                     "There are no unreviewed samples with high confidence to accept.")
        return
    folder = str(results_dir(win))

    def files_text(where: str) -> str:
        scope = st_review.to_write(win.state["results"], win.state["review"], False)
        scope = scope + [i for i in ids if i not in scope]
        return export_text(win, where, scope, True)[0]

    def pick(current: str) -> Optional[str]:
        return dialogs.choose_folder(win, "Choose the folder to save the results in", current)

    ok, save, folder = dialogs.accept_dialog(
        win, f"Accept {summary.plural(len(ids), 'sample')} with high confidence ({', '.join(ids)})?\n\n"
             "They will be recorded as accepted (bulk). Samples you have already accepted or rejected are "
             "not changed.", folder, files_text, pick)
    if not ok:
        return
    for sid in st_review.accept_bulk(win.state["results"], win.state["review"]):
        autosave_session(win, sid)
    refresh_review_items(win)
    if save:
        write_chosen(win, folder, st_review.to_write(win.state["results"], win.state["review"], False), True)


def refresh_review_list(win, select: Optional[str] = None) -> None:
    lst = win.s3.samples
    lst.blockSignals(True)
    lst.clear()
    for sid in win.state["results"]:
        item = QListWidgetItem()
        item.setData(Qt.UserRole, sid)
        fill_review_item(win, item, sid)
        lst.addItem(item)
    lst.blockSignals(False)
    if lst.count():
        ids = list(win.state["results"])
        lst.setCurrentRow(ids.index(select) if select in ids else 0)
    show_review(win)


def set_badge(label, level: str) -> None:
    label.setText(level.upper())
    label.setStyleSheet(f"background: {BADGE_COLORS.get(level, '#555555')}; color: white; font-weight: bold; "
                        "padding: 6px 10px; border-radius: 6px;")


def show_empty(win, text: str) -> None:
    win.s3.empty.setText(text)
    win.s3.views.setCurrentIndex(2)


def all_viewers(win) -> list:
    return [win.s3.viewer, win.s3.viewer_left, win.s3.viewer_right]


def show_review(win) -> None:
    s3 = win.s3
    win.review_stale = False
    win.flicker_timer.stop()
    sid = review_sample_id(win)
    res = win.state["results"].get(sid) if sid else None
    for b in (s3.export_one, s3.export_accepted):
        b.setEnabled(res is not None)
    refresh_status_row(win)
    s3.layer.blockSignals(True)
    s3.layer.clear()
    for v in all_viewers(win):
        v.clear()
    s3.caption.setText("")
    if res is None:
        s3.banner.setVisible(False)
        s3.overall.setText("")
        s3.onto.setText("")
        s3.ref_name.setText("")
        s3.badge.setText("")
        s3.badge.setStyleSheet("")
        s3.reason.setText("")
        s3.notes.setText("")
        s3.steps.text.setPlainText("")
        s3.details.text.setPlainText("")
        show_empty(win, "Run a sample or open a session to review it.")
        s3.layer.blockSignals(False)
        return
    comp = composed_result(win, sid)
    s3.ref_name.setText(summary.name(review_reference(win, sid) or ""))
    mods = ro.registered_modalities(comp) if res["status"] == "ok" else []
    s3.layer.addItems([summary.name(m) for m in mods])
    win.layer_keys = mods
    s3.layer.blockSignals(False)
    show_layer_text(win)
    if not mods:
        show_empty(win, "No registered layer to show for this sample.")
        return
    render_view(win)
    if current_mode(win) == "flicker":
        win.flicker_timer.start()


def current_modality(win) -> str:
    i = win.s3.layer.currentIndex()
    return win.layer_keys[i] if 0 <= i < len(win.layer_keys) else ""


def show_layer_text(win) -> None:
    """Confidence card for the selected layer and the Review reference (the
    steps that matter for it); the overall sample confidence is separate."""
    s3 = win.s3
    sid = review_sample_id(win)
    res = win.state["results"].get(sid) if sid else None
    if res is None:
        return
    modality = current_modality(win)
    ref = review_reference(win, sid) or ""
    s3.overall.setText(summary.overall_text(res))
    s3.onto.setText(f"Showing {summary.name(modality)} registered onto {summary.name(ref)}" if modality else "")
    steps = []
    if res["status"] != "ok":
        level = res["status"]
        reason = summary.failure_text(res)
    elif modality:
        steps = summary.layer_steps(res, modality, ref)
        level = summary.steps_level(res, steps)
        reason = summary.layer_headline(level, modality, ref)
        if level == "low":
            reason += " " + errors.LOW_ADVICE
    else:
        level, reason = res["confidence"], summary.failure_text(res)
    set_badge(s3.badge, level)
    s3.reason.setText(reason)
    s3.steps.text.setPlainText(summary.steps_text(steps))
    s3.steps.toggle.setVisible(bool(steps))
    low = level == "low" or (res["status"] == "ok" and res["confidence"] == "low")
    s3.banner.setVisible(low)
    if low:
        s3.banner.setText("LOW CONFIDENCE: check this registration carefully before using it. " + errors.LOW_ADVICE)
    raw_notes = list(res.get("warnings") or [])
    detail_lines = [summary.confidence_rule_text(res), f"review reference: {ref}"]
    for mod, layer in win.state["inputs"].get(sid, {}).items():
        desc, _err = summary.describe(layer)
        raw_notes += summary.descriptor_warnings(desc)
        detail_lines.append(summary.details_text(desc, layer))
    plain, _all = summary.split_notes(raw_notes)
    s3.notes.setText("Notes:\n- " + "\n- ".join(plain) if plain else "")
    s3.notes.setVisible(bool(plain))
    err = win.overlay_errors.get((sid, modality, ref))
    if err:
        detail_lines.append("Overlay error:\n" + err)
    s3.details.text.setPlainText("\n\n".join(detail_lines))


def on_layer_change(win) -> None:
    show_layer_text(win)
    render_view(win)


def current_mode(win) -> str:
    return next(k for k, b in win.s3.mode_buttons.items() if b.isChecked())


def color_on(win) -> bool:
    return win.s3.color_buttons["color"].isChecked() and current_mode(win) != "side"


def get_overlay(win, sid: str, modality: str):
    """Overlays are built in the Review reference's frame at display
    resolution, on the UI thread with a busy cursor and a "Building the view"
    message, then cached per sample, layer and reference."""
    key = (sid, modality, review_reference(win, sid))
    if key not in win.overlay_cache:
        show_empty(win, "Building the view...")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents(QEventLoop.ExcludeUserInputEvents)
        try:
            win.overlay_cache[key] = ro.build_review_images(
                win.state["inputs"][sid], composed_result(win, sid), modality, ro.DISPLAY_UM_PER_PX)
        finally:
            QApplication.restoreOverrideCursor()
    return win.overlay_cache[key]


def compose_current(win, imgs: dict):
    mode, color = current_mode(win), color_on(win)
    ref, moved = imgs["reference"], imgs["moved"]
    if mode == "blend":
        return ro.compose_blend(ref, moved, win.s3.slider.value() / 100.0, color=color)
    if mode == "checker":
        return ro.compose_checkerboard(ref, moved, color=color)
    return ro.compose_flicker_frame(ref, moved, win.flicker_on, color=color)


def render_view(win) -> None:
    s3 = win.s3
    sid, modality = review_sample_id(win), current_modality(win)
    if not sid or not modality:
        return
    ref = review_reference(win, sid)
    try:
        imgs = get_overlay(win, sid, modality)
    except Exception as exc:  # noqa: BLE001 - plain message shown, exception logged
        win.overlay_errors[(sid, modality, ref)] = errors.log_exception(f"overlay {sid} {modality} on {ref}", exc)
        for v in all_viewers(win):
            v.clear()
        s3.caption.setText("")
        show_empty(win, errors.OVERLAY_FAILED)
        show_layer_text(win)
        return
    if current_mode(win) == "side":
        left, right = ro.compose_side_panes(imgs["reference"], imgs["moved"], imgs.get("reference_rgb"),
                                            imgs.get("moved_rgb"))
        s3.viewer_left.set_image(left)
        s3.viewer_right.set_image(right)
        s3.pane_left.setText(f"{summary.name(ref)} (reference)")
        s3.pane_right.setText(f"{summary.name(modality)} (registered)")
        s3.views.setCurrentIndex(1)
        shown = [s3.viewer_left, s3.viewer_right]
    else:
        s3.viewer.set_image(compose_current(win, imgs))
        s3.views.setCurrentIndex(0)
        shown = [s3.viewer]
    for v in shown:
        v.set_markers(imgs["markers"])
        v.set_markers_visible(s3.markers.isChecked())
    caption = ""
    if imgs["n_markers"]:
        caption = (f"Showing {imgs['n_markers']} of {imgs['n_matched']} matched points used for the fit "
                   "(thinned for display)")
    s3.caption.setText(caption)
    s3.markers.setToolTip(caption)


def set_markers_visible(win, on: bool) -> None:
    for v in all_viewers(win):
        v.set_markers_visible(on)


def link_panes(win) -> None:
    """Zooming or panning either side-by-side pane moves the other; the
    cursor position is shown as a marker in both."""
    left, right = win.s3.viewer_left, win.s3.viewer_right
    left.view_changed.connect(lambda: right.set_view_state(left.view_state()))
    right.view_changed.connect(lambda: left.set_view_state(right.view_state()))
    for v in (left, right):
        v.cursor_moved.connect(lambda pt: (left.set_cursor_marker(pt), right.set_cursor_marker(pt)))


def on_mode_change(win) -> None:
    mode = current_mode(win)
    win.s3.slider.setEnabled(mode == "blend")
    for b in win.s3.color_buttons.values():
        b.setEnabled(mode != "side")
    if mode == "flicker":
        win.flicker_on = False
        win.flicker_timer.start()
    else:
        win.flicker_timer.stop()
    render_view(win)


def flicker_tick(win) -> None:
    win.flicker_on = not win.flicker_on
    render_view(win)



# ---------------------------------------------------------------- Export

def pixel_sizes_for(win, sid: str) -> dict:
    layers = win.state["inputs"].get(sid, {})
    return {m: (summary.describe(l)[0] or {}).get("pixel_size_um") for m, l in layers.items()}


def reference_for(win, sid: str) -> Optional[str]:
    return review_reference(win, sid)


SUMMARY_NAME = "batch_summary.csv"


def plan_samples(win, folder: str, ids: List[str]):
    """(plans, no_registration, notes) for the samples in ids, nothing written."""
    plans, skipped, notes = [], [], []
    for sid in ids:
        res = win.state["results"][sid]
        ref, sizes = reference_for(win, sid), pixel_sizes_for(win, sid)
        plan = session.plan_export(res, Path(folder) / sid, reference=ref, pixel_sizes=sizes)
        if not any(p["file"].name.endswith("_matrix.csv") or p["file"].name.endswith("_imagealignment.csv") for p in plan):
            skipped.append(sid)
            continue
        plans.append((sid, res, plan, ref, sizes))
        for n in session.export_notes(res, ref, sizes):
            if n not in notes:
                notes.append(n)
    return plans, skipped, notes


def session_ids(win) -> List[str]:
    """Every sample in the session: the sample list (blocked and never-run samples included), then any
    result whose sample is no longer in the list."""
    ids = [s["sample_id"] for s in win.state["samples"]]
    return ids + [sid for sid in win.state["results"] if sid not in ids]


def not_run_text(win, sid: str) -> str:
    """Plain status for a sample with no result, for example "not run: needs at least two linked layers"."""
    sample = st.find_sample(win.state, sid)
    reasons = st.blocked_reasons(sample) if sample is not None else []
    return "not run: " + "; ".join(reasons) if reasons else "not run yet"


def no_matrix_reasons(win, folder: str, scope: List[str]) -> Dict[str, str]:
    """{sample: plain reason} for the samples in scope that have no matrices to write."""
    out: Dict[str, str] = {}
    have = [sid for sid in scope if sid in win.state["results"]]
    _plans, skipped, _notes = plan_samples(win, folder, have)
    for sid in scope:
        if sid not in win.state["results"]:
            out[sid] = not_run_text(win, sid).replace("not run: ", "")
        elif sid in skipped:
            status = win.state["results"][sid].get("status")
            out[sid] = {"failed": "registration failed", "cancelled": "run cancelled"}.get(status, "no registration result")
    return out


def export_text(win, folder: str, chosen: List[str], batch: bool):
    """(text, plans, n_existing) describing an export of the samples in chosen."""
    plans, _skipped, notes = plan_samples(win, folder, chosen)
    lines, existing = [], 0
    for _sid, _res, plan, _ref, _sizes in plans:
        for p in plan:
            existing += bool(p["exists"])
            lines.append(f"{p['file']}  ({p['what']})" + ("  EXISTS" if p["exists"] else ""))
    if plans:
        summary_file = Path(folder) / SUMMARY_NAME
        existing += summary_file.exists()
        lines.append(f"{summary_file}  (one row per sample in this session: confidence, review status, export time)"
                     + ("  EXISTS" if summary_file.exists() else ""))
    text = f"Destination: {folder}\n\nThese files will be written:\n\n" + ("\n".join(lines) if lines else "(none)")
    if notes:
        text += "\n\n" + "\n".join(notes)
    nomat = no_matrix_reasons(win, folder, session_ids(win) if batch else list(chosen))
    if nomat:
        text += "\n\n" + "\n".join(f"Not written: no matrices ({reason}): {sid}" for sid, reason in nomat.items())
        if batch:
            text += "\nThe \"Include unreviewed and rejected samples\" option cannot write these."
    if batch:
        left = [f"{sid}: {review_label(win, sid)}" for sid in win.state["results"]
                if sid not in chosen and sid not in nomat]
        if left:
            text += "\n\nListed but not written (tick \"Include unreviewed and rejected samples\" to write them):\n"
            text += "\n".join(left)
        if not chosen:
            text += "\n\nNo sample is accepted yet. Accept samples on the Review screen first."
    return text, plans, existing


def write_summary(win, folder: str, written: dict) -> None:
    """batch_summary.csv: one row for every sample in the session, including blocked and never-run samples
    (blank confidence, status text as review status); export_time only for samples written by an export
    (earlier export times in an existing file are kept for samples not written this time)."""
    path = Path(folder) / SUMMARY_NAME
    earlier = session.read_export_times(path)
    rows = []
    for sid in session_ids(win):
        res = win.state["results"].get(sid)
        if res is None:
            sample = st.find_sample(win.state, sid)
            rows.append(session.not_run_row(sid, not_run_text(win, sid),
                                            st.chosen_reference(sample)[0] if sample else None, version.VERSION))
            continue
        rows.append(session.summary_row(res, reference_for(win, sid), review_label(win, sid),
                                        written.get(sid) or earlier.get(sid, ""), version.VERSION))
    session.write_batch_summary(rows, path)


def default_export_start(win) -> str:
    """Exports start in <output folder>/results (created if the output folder exists); the user can pick elsewhere."""
    if not output_folder(win).is_dir():
        return ""
    try:
        results_dir(win).mkdir(exist_ok=True)
    except OSError:
        return str(output_folder(win))
    return str(results_dir(win))


def export_label(win, sid: str) -> str:
    entry = export_entry(win, sid)
    if not entry:
        return "Not saved yet"
    try:
        when = datetime.fromisoformat(entry["time"]).strftime("%Y-%m-%d %H:%M")
    except (KeyError, ValueError):
        when = str(entry.get("time", ""))
    return f"Saved to {entry.get('folder', '')} at {when}"

def export_flow(win, ids: Optional[List[Optional[str]]]) -> None:
    """ids=[sample] exports that sample whatever its review status. ids=None is "Export accepted": accepted
    samples are written, the others are listed, and a tick box adds unreviewed and rejected ones."""
    batch = ids is None
    results = win.state["results"]
    chosen = [i for i in (ids or []) if i and i in results]
    if not results or (not batch and not chosen):
        dialogs.info(win, "Save", "No results to save.")
        return
    folder = dialogs.choose_folder(win, "Choose the folder to save the results in", default_export_start(win))
    if not folder:
        return
    if batch:
        def describe(inc):
            text, plans, _existing = export_text(win, folder, st_review.to_write(results, win.state["review"], inc), True)
            return text, bool(plans)

        ok, include = dialogs.export_dialog(win, describe)
        if not ok:
            return
        chosen = st_review.to_write(results, win.state["review"], include)
    write_chosen(win, folder, chosen, batch)


def write_chosen(win, folder: str, chosen: List[str], batch: bool) -> bool:
    """The save code path shared by "Save accepted results", "Save this sample's results" and the save box of
    "Accept all high-confidence". Asks before overwriting; writes the results and batch_summary.csv."""
    text, plans, existing = export_text(win, folder, chosen, batch)
    if not plans:
        dialogs.warn(win, "Save", "None of these samples has a registration to save.")
        return False
    if existing:
        text += (f"\n\n{summary.plural(existing, 'file')} already {'exists' if existing == 1 else 'exist'} "
                 "and would be overwritten. Overwrite?")
        proceed = dialogs.confirm(win, "Overwrite existing files?", text, default_yes=False)
    elif batch:
        proceed = True
    else:
        proceed = dialogs.confirm(win, "Save", text + "\n\nWrite them?", default_yes=True)
    if not proceed:
        return False
    stamp = datetime.now().isoformat(timespec="seconds")
    try:
        for sid, res, _plan, ref, sizes in plans:
            session.export_result(res, Path(folder) / sid, reference=ref, pixel_sizes=sizes)
        write_summary(win, folder, {sid: stamp for sid, *_rest in plans})
        for sid, *_rest in plans:
            win.state["exports"][sid] = {"time": stamp, "folder": str(Path(folder) / sid)}
            autosave_session(win, sid)
        refresh_status_row(win)
    except Exception as exc:  # noqa: BLE001 - plain message shown, exception logged
        errors.log_exception("export", exc)
        dialogs.warn(win, "Save", "The results could not be saved. Check that the folder can be "
                                  "written to. Technical detail is in the log file.")
        return False
    n_files = sum(len(plan) for _sid, _res, plan, _ref, _sizes in plans) + 1
    dialogs.info(win, "Save", f"Wrote {summary.plural(n_files, 'file')} under {folder}.")
    return True


# ---------------------------------------------------------------- File menu

def new_session(win) -> None:
    if is_running(win):
        dialogs.warn(win, "New", "A run is in progress. Cancel it first.")
        return
    if (win.state["samples"] or win.state["results"]) and not dialogs.confirm(
            win, "New", "Discard the current samples and results?", default_yes=False):
        return
    win.state = st.new_state()
    init_output_folder(win)
    win.overlay_cache.clear()
    win.overlay_errors.clear()
    summary.clear_cache()
    win.run_message = ""
    win.s1.bar.setValue(0)
    win.s1.step.setText("")
    win.s1.open_review.setVisible(False)
    win.s1.target.setValue(1.0)
    refresh_all(win)
    refresh_review_list(win)
    show_page(win, PAGE_DATA)


def save_current_session(win) -> None:
    sid = review_sample_id(win) if win.shell.pages.currentIndex() == PAGE_REVIEW else None
    if sid is None:
        sample = current_sample(win)
        sid = None if sample is None else sample["sample_id"]
    if sid is None or sid not in win.state["results"]:
        dialogs.warn(win, "Save session", "Select a sample that has been run to save its session.")
        return
    path = dialogs.choose_save_session(win, f"{sid}_session.json")
    if not path:
        return
    save_session_file(win, path, sid)
    dialogs.info(win, "Save session", f"Saved session for {sid}.")


def open_session(win) -> None:
    path = dialogs.choose_open_session(win)
    if not path:
        return
    try:
        data = session.load_session(path)
    except Exception as exc:  # noqa: BLE001 - plain message shown, exception logged
        errors.log_exception("open session", exc)
        dialogs.warn(win, "Open session", "That file could not be read as a session. Technical detail is in the log file.")
        return
    res = data["result"]
    sid = res["sample_id"]
    old = st.find_sample(win.state, sid)
    if old is not None:
        win.state["samples"].remove(old)
    sample = st.make_sample(sid)
    for modality, layer in data["inputs"].items():
        st.set_layer(sample, modality, layer["path"], pixel_size_um=layer.get("pixel_size_um"), channel=layer.get("channel"))
    sample["reference"] = data.get("reference")
    win.state["samples"].append(sample)
    win.state["results"][sid] = res
    win.state["review"][sid] = st_review.clean(data.get("review"))
    if isinstance(data.get("export"), dict):
        win.state["exports"][sid] = data["export"]
    else:
        win.state["exports"].pop(sid, None)
    win.state["inputs"][sid] = copy.deepcopy(sample["layers"])
    win.overlay_cache.clear()
    summary.clear_cache()
    loading.read_with_progress(win, [(sid, list(sample["layers"].values()))], "Opening session")
    refresh_all(win)
    refresh_review_list(win, select=sid)
    show_page(win, PAGE_REVIEW)


def run_app(argv) -> int:
    app = QApplication.instance() or QApplication(argv)
    theme.apply(app)
    win = MainWindow()
    win.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(run_app(sys.argv))


