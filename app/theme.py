"""One Qt stylesheet for the whole app (no extra dependencies).

Colors: deep navy header and text, teal primary buttons, light teal
highlights, off-white content background, white cards with 1 px light gray
borders and 8 px corners. Light theme only. Widgets pick a look from the
"role" dynamic property: primary, secondary (default for buttons), link,
step, segment; cards use objectName "card". The Review screen's
confidence badge keeps its own inline color.
"""
from __future__ import annotations

NAVY = "#0C2B45"
TEAL = "#0F6E56"
TEAL_HOVER = "#0B5844"
LIGHT_TEAL = "#5DCAA5"
OFF_WHITE = "#F4F8FC"
WHITE = "#FFFFFF"
BORDER = "#D5DCE4"
MUTED = "#5B6B7B"
AMBER = "#D98E04"
RED = "#C0392B"
GREEN = "#2E9E6B"
GRAY_DOT = "#9AA5B1"
FONT_PT = 10.5
FONT_FAMILIES = ["Segoe UI", "Helvetica Neue", "Arial", "sans-serif"]


def stylesheet() -> str:
    return f"""
* {{ color: {NAVY}; }}
QMainWindow, QWidget#content, QScrollArea, QScrollArea > QWidget > QWidget {{ background: {OFF_WHITE}; }}
QWidget {{ font-size: {FONT_PT}pt; }}
QLabel {{ background: transparent; }}
QLabel[role="muted"] {{ color: {MUTED}; }}
QLabel[role="title"] {{ font-size: 13pt; font-weight: 600; }}
QLabel[role="warn"] {{ color: #8A5A00; }}
QLabel[role="block"] {{ color: {RED}; }}

QFrame#header {{ background: {NAVY}; border: none; }}
QFrame#header QLabel {{ color: {WHITE}; }}
QLabel#appname {{ font-size: 13pt; font-weight: 600; color: {WHITE}; }}
QPushButton[role="step"] {{ background: transparent; color: {WHITE}; border: none; border-bottom: 3px solid transparent;
    padding: 10px 14px; border-radius: 0; font-weight: 600; }}
QPushButton[role="step"]:hover:enabled {{ border-bottom: 3px solid {LIGHT_TEAL}; }}
QPushButton[role="step"][active="true"] {{ border-bottom: 3px solid {LIGHT_TEAL}; }}
QPushButton[role="step"]:disabled {{ color: #6F8499; }}

QFrame#card, QFrame#panel {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 8px; }}
QFrame#card QLabel, QFrame#panel QLabel {{ background: transparent; }}
QFrame#card[state="empty"] {{ border: 2px dashed #A9B8C7; background: {WHITE}; }}
QFrame#card[state="over"], QFrame#dropzone[state="over"] {{ border: 2px solid {LIGHT_TEAL}; background: #EAF8F3; }}
QFrame#card[state="warn"] {{ border: 1px solid {AMBER}; }}
QFrame#card[state="block"] {{ border: 1px solid {RED}; }}
QFrame#dropzone {{ background: {WHITE}; border: 3px dashed #A9B8C7; border-radius: 12px; }}
QFrame#footer {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 8px; }}

QPushButton, QToolButton {{ background: {WHITE}; color: {NAVY}; border: 1px solid {BORDER}; border-radius: 6px;
    padding: 6px 14px; }}
QPushButton:hover:enabled, QToolButton:hover:enabled {{ background: #EAF2F8; border-color: #B7C5D3; }}
QPushButton:pressed, QToolButton:pressed {{ background: #DCE8F1; }}
QPushButton:disabled, QToolButton:disabled {{ color: #9AA8B6; background: #F1F4F7; border-color: #E4E9EE; }}
QToolButton::menu-indicator {{ image: none; width: 0; }}
QPushButton[role="primary"] {{ background: {TEAL}; color: {WHITE}; border: 1px solid {TEAL}; font-weight: 600;
    padding: 8px 18px; }}
QPushButton[role="primary"]:hover:enabled {{ background: {TEAL_HOVER}; border-color: {TEAL_HOVER}; }}
QPushButton[role="primary"]:disabled {{ background: #A9CFC4; color: {WHITE}; border-color: #A9CFC4; }}
QPushButton[role="link"], QToolButton[role="link"] {{ background: transparent; border: none; color: {TEAL};
    padding: 2px 4px; text-decoration: underline; }}
QPushButton[role="link"]:hover:enabled, QToolButton[role="link"]:hover:enabled {{ background: transparent; color: {TEAL_HOVER}; }}
QPushButton[role="link"]:disabled {{ background: transparent; color: #9AA8B6; border: none; }}
QPushButton[role="segment"] {{ border-radius: 0; margin: 0; padding: 7px 18px; border: 1px solid {BORDER}; }}
QPushButton[role="segment"]:checked {{ background: {TEAL}; color: {WHITE}; border-color: {TEAL}; font-weight: 600; }}
QPushButton[role="segment"]:disabled {{ color: #9AA8B6; }}
QPushButton[role="segment"][pos="left"] {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
QPushButton[role="segment"][pos="right"] {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; }}
QPushButton[role="segment"][compact="true"] {{ padding: 7px 6px; }}
QLabel[role="badge"] {{ color: {WHITE}; font-weight: 700; padding: 6px 10px; border-radius: 6px; }}
QLabel[role="banner"], QFrame#card QLabel[role="banner"] {{ background: {RED}; color: {WHITE}; font-weight: 600; padding: 8px; border-radius: 6px; }}
QPlainTextEdit[role="steps"] {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 6px; }}

QLineEdit, QDoubleSpinBox, QComboBox {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 5px 8px;
    selection-background-color: {LIGHT_TEAL}; selection-color: {NAVY}; }}
QLineEdit:focus, QDoubleSpinBox:focus, QComboBox:focus {{ border: 1px solid {TEAL}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: {WHITE}; border: 1px solid {BORDER}; selection-background-color: {LIGHT_TEAL};
    selection-color: {NAVY}; }}
QPlainTextEdit {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 6px; padding: 4px; }}

QListWidget {{ background: {WHITE}; border: 1px solid {BORDER}; border-radius: 8px; padding: 4px; outline: none; }}
QListWidget::item {{ padding: 8px 6px; border-radius: 6px; }}
QListWidget::item:hover {{ background: #EAF2F8; }}
QListWidget::item:selected {{ background: #DDF3EB; color: {NAVY}; border: 1px solid {LIGHT_TEAL}; }}

QProgressBar {{ background: #E3EAF1; border: none; border-radius: 6px; min-height: 12px; max-height: 12px; text-align: center;
    color: transparent; }}
QProgressBar::chunk {{ background: {TEAL}; border-radius: 6px; }}

QSlider::groove:horizontal {{ height: 4px; background: {BORDER}; border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {TEAL}; width: 14px; margin: -6px 0; border-radius: 7px; }}
QRadioButton, QCheckBox {{ spacing: 6px; }}
QToolButton[role="toggle"] {{ background: transparent; border: none; padding: 2px 4px; color: {MUTED}; }}

QScrollBar:vertical {{ background: transparent; width: 10px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #C3CED9; border-radius: 5px; min-height: 24px; }}
QScrollBar:horizontal {{ background: transparent; height: 10px; }}
QScrollBar::handle:horizontal {{ background: #C3CED9; border-radius: 5px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}

QMenuBar {{ background: {WHITE}; border-bottom: 1px solid {BORDER}; }}
QMenuBar::item:selected {{ background: #EAF2F8; }}
QMenu {{ background: {WHITE}; border: 1px solid {BORDER}; padding: 4px; }}
QMenu::item {{ padding: 6px 22px; border-radius: 4px; }}
QMenu::item:selected {{ background: #DDF3EB; }}
QToolTip {{ background: {NAVY}; color: {WHITE}; border: none; padding: 4px 6px; }}
"""


def apply(app) -> None:
    """Apply the stylesheet and the default font app-wide."""
    from PyQt5.QtGui import QFont
    font = QFont()
    font.setFamilies(FONT_FAMILIES)
    font.setPointSizeF(FONT_PT)
    app.setFont(font)
    app.setStyleSheet(stylesheet())


def repolish(widget) -> None:
    """Re-evaluate the stylesheet after a dynamic property changed."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def set_prop(widget, name: str, value) -> None:
    widget.setProperty(name, value)
    repolish(widget)
