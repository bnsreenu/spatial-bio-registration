"""Modal dialogs as plain functions. The rest of the app calls them as
dialogs.name(...) at call time, so tests can replace them."""
from __future__ import annotations

from typing import Optional

from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog, QHBoxLayout, QInputDialog, QLabel, QLineEdit, QMessageBox,
    QPlainTextEdit, QPushButton, QVBoxLayout,
)


def choose_folder(parent, title: str, start: str = "") -> Optional[str]:
    path = QFileDialog.getExistingDirectory(parent, title, start)
    return path or None


def choose_file(parent, title: str) -> Optional[str]:
    path, _ = QFileDialog.getOpenFileName(parent, title)
    return path or None


def choose_files(parent, title: str) -> list:
    paths, _ = QFileDialog.getOpenFileNames(parent, title)
    return list(paths)


def choose_open_session(parent) -> Optional[str]:
    path, _ = QFileDialog.getOpenFileName(parent, "Open session", "", "Session (*.json)")
    return path or None


def choose_save_session(parent, default_name: str) -> Optional[str]:
    path, _ = QFileDialog.getSaveFileName(parent, "Save session", default_name, "Session (*.json)")
    return path or None


def ask_pixel_size(parent, label: str) -> Optional[float]:
    value, ok = QInputDialog.getDouble(
        parent, "Pixel size needed",
        f"{label}: the file has no pixel size.\nEnter it in um per pixel (it is not guessed):",
        0.0, 0.0, 1000.0, 6,
    )
    return float(value) if ok and value > 0 else None


def rerun_dialog(parent, text: str, offer_all: bool) -> tuple:
    """Ask whether to re-run a sample that already has a result. Returns (rerun, apply_to_all).
    Closing the dialog counts as Skip."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("Re-run sample?")
    lay = QVBoxLayout(dlg)
    body = QLabel(text)
    body.setWordWrap(True)
    body.setMinimumWidth(420)
    lay.addWidget(body)
    every = QCheckBox("Apply to all remaining samples that already have a result")
    every.setVisible(offer_all)
    lay.addWidget(every)
    buttons = QDialogButtonBox()
    rerun = buttons.addButton("Re-run", QDialogButtonBox.AcceptRole)
    skip = buttons.addButton("Skip", QDialogButtonBox.RejectRole)
    skip.setDefault(True)
    rerun.setAutoDefault(False)
    lay.addWidget(buttons)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    return dlg.exec_() == QDialog.Accepted, every.isChecked()

def confirm(parent, title: str, text: str, default_yes: bool = True) -> bool:
    # A plain QDialog (not QMessageBox) so long lists scroll and the dialog can be driven in headless tests.
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    layout = QVBoxLayout(dlg)
    if len(text) > 400 or text.count("\n") > 8:
        body = QPlainTextEdit()
        body.setReadOnly(True)
        body.setPlainText(text)
        dlg.resize(680, 420)
    else:
        body = QLabel(text)
        body.setWordWrap(True)
        body.setMinimumWidth(380)
    layout.addWidget(body, 1)
    buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    (buttons.button(QDialogButtonBox.Yes) if default_yes else buttons.button(QDialogButtonBox.No)).setDefault(True)
    layout.addWidget(buttons)
    return dlg.exec_() == QDialog.Accepted


def show_text(parent, title: str, text: str) -> None:
    dlg = QDialog(parent)
    dlg.setWindowTitle(title)
    dlg.resize(760, 420)
    layout = QVBoxLayout(dlg)
    body = QPlainTextEdit()
    body.setReadOnly(True)
    body.setLineWrapMode(QPlainTextEdit.NoWrap)
    body.setPlainText(text)
    layout.addWidget(body, 1)
    buttons = QDialogButtonBox(QDialogButtonBox.Close)
    buttons.rejected.connect(dlg.reject)
    layout.addWidget(buttons)
    dlg.exec_()


def accept_dialog(parent, text: str, folder: str, files_text, pick_folder) -> tuple:
    """Confirm "Accept all high-confidence" with a save box. files_text(folder) -> text listing the files;
    pick_folder(current) -> a new folder or None. Returns (accepted, save_too, folder)."""
    from PyQt5.QtCore import Qt
    dlg = QDialog(parent)
    dlg.setWindowTitle("Accept all high-confidence")
    layout = QVBoxLayout(dlg)
    body = QLabel(text)
    body.setWordWrap(True)
    body.setMinimumWidth(460)
    layout.addWidget(body)
    save = QCheckBox()
    save.setChecked(True)
    state = {"folder": folder}
    path_label = QLabel()
    path_label.setWordWrap(True)
    path_label.setProperty("role", "muted")
    links = QLabel('<a href="change">Change...</a> &nbsp;&nbsp; <a href="show">Show files</a>')
    links.setTextInteractionFlags(Qt.TextBrowserInteraction)

    def refresh():
        save.setText(f"Also save their results to {state['folder']}")
        links.setEnabled(save.isChecked())

    def on_link(which):
        if which == "change":
            new = pick_folder(state["folder"])
            if new:
                state["folder"] = new
                refresh()
        else:
            show_text(dlg, "Files that will be written", files_text(state["folder"]))

    links.linkActivated.connect(on_link)
    save.toggled.connect(lambda _on: refresh())
    refresh()
    layout.addWidget(save)
    layout.addWidget(links)
    buttons = QDialogButtonBox(QDialogButtonBox.Yes | QDialogButtonBox.No)
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    buttons.button(QDialogButtonBox.No).setDefault(True)
    layout.addWidget(buttons)
    ok = dlg.exec_() == QDialog.Accepted
    return ok, save.isChecked(), state["folder"]


def show_dialog(dlg) -> None:
    dlg.exec_()


def export_dialog(parent, describe) -> tuple:
    """Show what an export would write. describe(include_all) -> (text, can_write) is called again when the
    "include unreviewed and rejected" box changes (default off). Returns (accepted, include_all)."""
    dlg = QDialog(parent)
    dlg.setWindowTitle("Save accepted results")
    dlg.resize(760, 520)
    layout = QVBoxLayout(dlg)
    include = QCheckBox("Include unreviewed and rejected samples")
    body = QPlainTextEdit()
    body.setReadOnly(True)
    body.setLineWrapMode(QPlainTextEdit.NoWrap)
    buttons = QDialogButtonBox(QDialogButtonBox.Cancel)
    write = buttons.addButton("Write files", QDialogButtonBox.AcceptRole)
    layout.addWidget(include)
    layout.addWidget(body, 1)
    layout.addWidget(buttons)

    def refresh():
        text, can_write = describe(include.isChecked())
        body.setPlainText(text)
        write.setEnabled(can_write)

    include.toggled.connect(lambda _on: refresh())
    buttons.accepted.connect(dlg.accept)
    buttons.rejected.connect(dlg.reject)
    refresh()
    ok = dlg.exec_() == QDialog.Accepted
    return ok, include.isChecked()


def info(parent, title: str, text: str) -> None:
    QMessageBox.information(parent, title, text)


def warn(parent, title: str, text: str) -> None:
    QMessageBox.warning(parent, title, text)
