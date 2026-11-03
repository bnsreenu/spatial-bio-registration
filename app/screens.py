"""Builders for the window shell, the Data screen and the Review screen.
Each returns a SimpleNamespace of the widgets the controller functions in
main_window.py need. Looks come from app/theme.py (dynamic "role"
properties), not from inline styles, except for status colors."""
from __future__ import annotations

from types import SimpleNamespace

from PyQt5.QtCore import QPointF, Qt
from PyQt5.QtGui import QColor, QIcon, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QAction, QButtonGroup, QCheckBox, QComboBox, QDoubleSpinBox, QFrame, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QListWidget, QMenu, QPlainTextEdit, QProgressBar, QPushButton, QRadioButton, QScrollArea,
    QSizePolicy, QSlider, QStackedWidget, QToolButton, QVBoxLayout, QWidget,
)

from app import theme
from app.widgets import DropFrame, ImageViewer

REVIEW_MODES = [("blend", "Blend"), ("checker", "Checkerboard"), ("side", "Side by side"), ("flicker", "Flicker")]
CARD_SPECS = [("he", "H&E"), ("comet", "COMET"), ("xenium", "Xenium output folder"), ("other", "Other (optional)")]
REFERENCE_SEGMENTS = [("xenium", "Xenium"), ("comet", "COMET"), ("he", "H&E")]
STEP_NAMES = ["1  Data", "2  Review and save"]


def button(text: str, role: str = "secondary") -> QPushButton:
    b = QPushButton(text)
    b.setProperty("role", role)
    return b


def label(text: str = "", role: str = None, wrap: bool = True) -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(wrap)
    if role:
        lab.setProperty("role", role)
    return lab


def dot_icon(color: str, size: int = 14) -> QIcon:
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.end()
    return QIcon(pix)


def status_pixmap(kind: str, size: int = 20) -> QPixmap:
    """A drawn check (kind "ok"), warning mark ("warn") or stop mark
    ("block"), so no icon font is needed."""
    color = {"ok": theme.GREEN, "warn": theme.AMBER, "block": theme.RED, "info": "#5B7C99"}[kind]
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)
    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(color))
    p.drawEllipse(1, 1, size - 2, size - 2)
    p.setPen(QPen(QColor("white"), 2.0, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
    s = size / 20.0
    if kind == "ok":
        p.drawPolyline(*[QPointF(x * s, y * s) for x, y in ((5.5, 10.5), (8.5, 13.5), (14.5, 6.5))])
    elif kind == "info":
        p.drawPoint(int(10 * s), int(5.5 * s))
        p.drawLine(int(10 * s), int(9 * s), int(10 * s), int(15 * s))
    else:
        p.drawLine(int(10 * s), int(5 * s), int(10 * s), int(11 * s) if kind == "warn" else int(15 * s))
        if kind == "warn":
            p.drawPoint(int(10 * s), int(15 * s))
    p.end()
    return pix


def make_details(layout, title: str = "Details") -> SimpleNamespace:
    """A collapsed section (toggle button + read-only text)."""
    toggle = QToolButton()
    toggle.setText(title)
    toggle.setProperty("role", "toggle")
    toggle.setCheckable(True)
    toggle.setArrowType(Qt.RightArrow)
    toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    text = QPlainTextEdit()
    text.setReadOnly(True)
    text.setMaximumHeight(170)
    text.setVisible(False)

    def on_toggle(on):
        text.setVisible(on)
        toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)

    toggle.toggled.connect(on_toggle)
    if layout is not None:
        layout.addWidget(toggle, 0, Qt.AlignLeft)
        layout.addWidget(text)
    return SimpleNamespace(toggle=toggle, text=text)


def file_menu_button(text: str, role: str):
    """A button with a small menu: "File..." and "Zarr folder..."."""
    btn = QToolButton()
    btn.setText(text)
    btn.setProperty("role", role)
    btn.setPopupMode(QToolButton.InstantPopup)
    menu = QMenu(btn)
    act_file = QAction("File...", btn)
    act_zarr = QAction("Zarr folder...", btn)
    menu.addAction(act_file)
    menu.addAction(act_zarr)
    btn.setMenu(menu)
    return btn, act_file, act_zarr


# ---------------------------------------------------------------- shell

def build_shell() -> SimpleNamespace:
    root = QWidget()
    root.setObjectName("content")
    lay = QVBoxLayout(root)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(0)
    header = QFrame()
    header.setObjectName("header")
    header.setFixedHeight(46)
    hl = QHBoxLayout(header)
    hl.setContentsMargins(16, 0, 16, 0)
    hl.setSpacing(4)
    name = QLabel("Spatial registration")
    name.setObjectName("appname")
    hl.addWidget(name)
    hl.addSpacing(28)
    steps = []
    for text in STEP_NAMES:
        b = QPushButton(text)
        b.setProperty("role", "step")
        b.setProperty("active", False)
        b.setCursor(Qt.PointingHandCursor)
        hl.addWidget(b)
        steps.append(b)
    hl.addStretch(1)
    pages = QStackedWidget()
    lay.addWidget(header)
    lay.addWidget(pages, 1)
    return SimpleNamespace(widget=root, header=header, steps=steps, pages=pages)


# ---------------------------------------------------------------- Data screen

def build_card(modality: str, title: str) -> SimpleNamespace:
    frame = DropFrame()
    frame.setObjectName("card")
    frame.setProperty("state", "empty")
    frame.setMinimumHeight(290)
    frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
    lay = QVBoxLayout(frame)
    lay.setContentsMargins(14, 12, 14, 12)
    lay.setSpacing(8)
    title_label = label(title, "title", wrap=False)
    lay.addWidget(title_label)
    stack = QStackedWidget()
    lay.addWidget(stack, 1)

    is_xenium = modality == "xenium"
    empty = QWidget()
    el = QVBoxLayout(empty)
    el.setContentsMargins(0, 0, 0, 0)
    el.addStretch(1)
    hint = label("Drop a folder here, or Browse" if is_xenium else "Drop a file or folder here, or Browse", "muted")
    hint.setAlignment(Qt.AlignCenter)
    el.addWidget(hint)
    if is_xenium:
        browse = button("Browse")
        browse_file = browse_zarr = None
    else:
        browse, browse_file, browse_zarr = file_menu_button("Browse", "secondary")
    el.addWidget(browse, 0, Qt.AlignCenter)
    cand_label = label("", "warn")
    cand_label.setAlignment(Qt.AlignCenter)
    cand_label.setVisible(False)
    cand_button = button("Pick one", "primary")
    cand_menu = QMenu(cand_button)
    cand_button.setMenu(cand_menu)
    cand_button.setVisible(False)
    el.addWidget(cand_label)
    el.addWidget(cand_button, 0, Qt.AlignCenter)
    el.addStretch(1)
    stack.addWidget(empty)

    loaded = QWidget()
    ll = QVBoxLayout(loaded)
    ll.setContentsMargins(0, 0, 0, 0)
    ll.setSpacing(6)
    top = QHBoxLayout()
    thumb = QLabel()
    thumb.setFixedSize(96, 96)
    thumb.setAlignment(Qt.AlignCenter)
    thumb.setStyleSheet("background: #EEF3F8; border-radius: 6px;")
    top.addWidget(thumb, 0, Qt.AlignTop)
    info = QVBoxLayout()
    name_row = QHBoxLayout()
    icon = QLabel()
    icon.setFixedSize(20, 20)
    name = QLabel("")
    name.setStyleSheet("font-weight: 600;")
    name_row.addWidget(icon)
    name_row.addWidget(name, 1)
    facts = label("", "muted")
    files = label("", "muted")
    info.addLayout(name_row)
    info.addWidget(facts)
    info.addStretch(1)
    top.addLayout(info, 1)
    ll.addLayout(top)
    ll.addWidget(files)
    msg = label("")
    msg.setVisible(False)
    ll.addWidget(msg)
    note_row = QWidget()
    nl = QHBoxLayout(note_row)
    nl.setContentsMargins(0, 0, 0, 0)
    note_icon = QLabel()
    note_icon.setPixmap(status_pixmap("info", 16))
    note_icon.setFixedSize(16, 16)
    note = label("", "muted")
    nl.addWidget(note_icon, 0, Qt.AlignTop)
    nl.addWidget(note, 1)
    note_row.setVisible(False)
    ll.addWidget(note_row)
    px_row = QWidget()
    pl = QHBoxLayout(px_row)
    pl.setContentsMargins(0, 0, 0, 0)
    px_spin = QDoubleSpinBox()
    px_spin.setDecimals(6)
    px_spin.setRange(0.0, 1000.0)
    px_spin.setSuffix(" um/px")
    px_set = button("Use")
    pl.addWidget(QLabel("Pixel size:"))
    pl.addWidget(px_spin, 1)
    pl.addWidget(px_set)
    px_row.setVisible(False)
    ll.addWidget(px_row)
    links = QHBoxLayout()
    if is_xenium:
        replace = button("Replace", "link")
        replace_file = replace_zarr = None
    else:
        replace, replace_file, replace_zarr = file_menu_button("Replace", "link")
    remove = button("Remove", "link")
    links.addWidget(replace)
    links.addWidget(remove)
    links.addStretch(1)
    ll.addLayout(links)
    details = make_details(ll)
    stack.addWidget(loaded)

    return SimpleNamespace(
        modality=modality, frame=frame, title=title_label, stack=stack, hint=hint, browse=browse,
        cand_label=cand_label, cand_button=cand_button, cand_menu=cand_menu,
        browse_file=browse_file, browse_zarr=browse_zarr, thumb=thumb, icon=icon, name=name, facts=facts,
        files=files, msg=msg, note_row=note_row, note_icon=note_icon, note=note, px_row=px_row,
        px_spin=px_spin, px_set=px_set, replace=replace,
        replace_file=replace_file, replace_zarr=replace_zarr, remove=remove, details=details,
    )


def build_data_screen() -> SimpleNamespace:
    stack = QStackedWidget()

    # page 0: empty state, a large centered panel
    empty = QWidget()
    el = QVBoxLayout(empty)
    el.setContentsMargins(40, 40, 40, 40)
    el.addStretch(1)
    drop = DropFrame()
    drop.setObjectName("dropzone")
    drop.setMinimumHeight(240)
    dl = QVBoxLayout(drop)
    dl.setAlignment(Qt.AlignCenter)
    drop_text = label("Drop a folder with your H&E, COMET and Xenium data here", "title")
    drop_text.setAlignment(Qt.AlignCenter)
    dl.addWidget(drop_text)
    brow = QHBoxLayout()
    brow.addStretch(1)
    empty_folder = button("Choose folder", "primary")
    empty_add = button("Add sample")
    brow.addWidget(empty_folder)
    brow.addWidget(empty_add)
    brow.addStretch(1)
    dl.addSpacing(12)
    dl.addLayout(brow)
    el.addWidget(drop)
    el.addStretch(1)
    stack.addWidget(empty)

    # page 1: sidebar + sample editor + footer
    page = QWidget()
    outer = QVBoxLayout(page)
    outer.setContentsMargins(16, 16, 16, 16)
    outer.setSpacing(14)
    body = QHBoxLayout()
    body.setSpacing(16)

    side = QFrame()
    side.setObjectName("panel")
    side.setFixedWidth(270)
    sl = QVBoxLayout(side)
    sl.setContentsMargins(12, 12, 12, 12)
    sl.setSpacing(8)
    sl.addWidget(label("Samples", "title", wrap=False))
    samples = QListWidget()
    sl.addWidget(samples, 1)
    add_sample = button("Add sample")
    add_folder = button("Add folder of samples")
    remove_sample = button("Remove")
    for b in (add_sample, add_folder, remove_sample):
        sl.addWidget(b)
    sl.addSpacing(6)
    sl.addWidget(label("Output folder", "title", wrap=False))
    out_path = label("", "muted")
    out_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
    out_change = button("Change...")
    out_note = label("Sessions are saved in sessions/, saved results in results/.", "muted")
    sl.addWidget(out_path)
    sl.addWidget(out_change)
    sl.addWidget(out_note)
    body.addWidget(side)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    inner = QWidget()
    inner.setObjectName("content")
    il = QVBoxLayout(inner)
    il.setContentsMargins(0, 0, 8, 0)
    il.setSpacing(14)
    name_row = QHBoxLayout()
    name_row.addWidget(label("Sample name", "muted", wrap=False))
    sample_name = QLineEdit()
    sample_name.setMinimumWidth(260)
    name_row.addWidget(sample_name)
    proposed = label("", "warn")
    name_row.addWidget(proposed, 1)
    il.addLayout(name_row)

    grid = QGridLayout()
    grid.setSpacing(14)
    cards = {}
    for i, (modality, title) in enumerate(CARD_SPECS):
        card = build_card(modality, title)
        cards[modality] = card
        grid.addWidget(card.frame, i // 2, i % 2)
    grid.setColumnStretch(0, 1)
    grid.setColumnStretch(1, 1)
    il.addLayout(grid)

    ref_panel = QFrame()
    ref_panel.setObjectName("panel")
    ref_panel.setMinimumHeight(130)
    rl = QVBoxLayout(ref_panel)
    rl.setContentsMargins(14, 12, 14, 12)
    rl.setSpacing(6)
    rrow = QHBoxLayout()
    rrow.addWidget(label("Register everything onto:", wrap=False))
    seg_group = QButtonGroup(ref_panel)
    seg_group.setExclusive(True)
    segments = {}
    for i, (modality, text) in enumerate(REFERENCE_SEGMENTS):
        b = QPushButton(text.replace("&", "&&"))
        b.setCheckable(True)
        b.setProperty("role", "segment")
        b.setProperty("pos", "left" if i == 0 else "right" if i == len(REFERENCE_SEGMENTS) - 1 else "mid")
        seg_group.addButton(b)
        segments[modality] = b
        rrow.addWidget(b)
    rrow.addStretch(1)
    rl.addLayout(rrow)
    ref_reason = label("", "muted")
    rl.addWidget(ref_reason)
    adv_toggle = QToolButton()
    adv_toggle.setText("Advanced")
    adv_toggle.setProperty("role", "toggle")
    adv_toggle.setCheckable(True)
    adv_toggle.setArrowType(Qt.RightArrow)
    adv_toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
    adv_panel = QWidget()
    al = QHBoxLayout(adv_panel)
    al.setContentsMargins(0, 0, 0, 0)
    al.addWidget(QLabel("Target resolution (um/px):"))
    target = QDoubleSpinBox()
    target.setDecimals(2)
    target.setRange(0.5, 4.0)
    target.setSingleStep(0.25)
    target.setValue(1.0)
    al.addWidget(target)
    al.addWidget(label("Default 1.0, the only value tested.", "muted"), 1)
    adv_panel.setVisible(False)
    rl.addWidget(adv_toggle, 0, Qt.AlignLeft)
    rl.addWidget(adv_panel)
    il.addWidget(ref_panel)
    il.addStretch(1)
    scroll.setWidget(inner)
    body.addWidget(scroll, 1)
    outer.addLayout(body, 1)

    footer = QFrame()
    footer.setObjectName("footer")
    fl = QHBoxLayout(footer)
    fl.setContentsMargins(14, 10, 14, 10)
    fl.setSpacing(12)
    ftext = QVBoxLayout()
    status = label("")
    step = label("", "muted")
    step.setVisible(False)
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setTextVisible(False)
    bar.setVisible(False)
    ftext.addWidget(status)
    ftext.addWidget(step)
    ftext.addWidget(bar)
    fl.addLayout(ftext, 1)
    cancel = button("Cancel")
    cancel.setVisible(False)
    open_review = button("Review results", "primary")
    open_review.setVisible(False)
    run_all = button("Auto-register all ready samples")
    run = button("Run auto-registration", "primary")
    for b in (cancel, open_review, run_all, run):
        fl.addWidget(b)
    outer.addWidget(footer)
    stack.addWidget(page)

    return SimpleNamespace(
        widget=stack, stack=stack, drop=drop, drop_text=drop_text, empty_folder=empty_folder, empty_add=empty_add,
        samples=samples, add_sample=add_sample, add_folder=add_folder, remove_sample=remove_sample,
        out_path=out_path, out_change=out_change, out_note=out_note,
        sample_name=sample_name, proposed=proposed, cards=cards, segments=segments, seg_group=seg_group,
        ref_reason=ref_reason, adv_toggle=adv_toggle, adv_panel=adv_panel, target=target,
        status=status, step=step, bar=bar, cancel=cancel, open_review=open_review, run_all=run_all, run=run,
        footer=footer,
    )


# ---------------------------------------------------------------- Review screen

PANEL_WIDTH = 390


def build_help_dialog(parent=None):
    """Quick guide dialog: topic list on the left, text on the right. Returns
    (dialog, topic list, text browser); the caller shows it."""
    from PyQt5.QtWidgets import QDialog, QTextBrowser
    from app import help_text
    dlg = QDialog(parent)
    dlg.setWindowTitle("Quick guide")
    dlg.resize(860, 560)
    lay = QHBoxLayout(dlg)
    topics = QListWidget()
    topics.setFixedWidth(230)
    text = QTextBrowser()
    for title, _body in help_text.TOPICS:
        topics.addItem(title)
    lay.addWidget(topics)
    lay.addWidget(text, 1)

    def show(row):
        if 0 <= row < len(help_text.TOPICS):
            title, body = help_text.TOPICS[row]
            text.setPlainText(f"{title}\n\n{body}")

    topics.currentRowChanged.connect(show)
    topics.setCurrentRow(0)
    return dlg, topics, text


def segment_group(parent, items, checked_key) -> tuple:
    """Exclusive, checkable segmented buttons. Returns ({key: button}, group, layout)."""
    group = QButtonGroup(parent)
    group.setExclusive(True)
    row = QHBoxLayout()
    row.setSpacing(0)
    buttons = {}
    for i, (key, text) in enumerate(items):
        b = QPushButton(text.replace("&", "&&"))
        b.setCheckable(True)
        b.setProperty("role", "segment")
        b.setProperty("compact", True)
        b.setProperty("pos", "left" if i == 0 else "right" if i == len(items) - 1 else "mid")
        group.addButton(b)
        row.addWidget(b, 1)
        buttons[key] = b
    buttons[checked_key].setChecked(True)
    return buttons, group, row


def build_review_screen() -> SimpleNamespace:
    root = QWidget()
    outer = QHBoxLayout(root)
    outer.setContentsMargins(16, 16, 16, 16)
    outer.setSpacing(14)

    samples = QListWidget()
    left = QVBoxLayout()
    left.setSpacing(8)
    left.addWidget(samples, 1)
    accept_high = button("Accept all high-confidence")
    left.addWidget(accept_high)
    left_box = QWidget()
    left_box.setFixedWidth(210)
    left_box.setLayout(left)
    left.setContentsMargins(0, 0, 0, 0)
    outer.addWidget(left_box)

    # center: one viewer for the overlay modes, two linked viewers for side by side
    views = QStackedWidget()
    viewer = ImageViewer()
    side = QWidget()
    sl = QHBoxLayout(side)
    sl.setContentsMargins(0, 0, 0, 0)
    sl.setSpacing(8)
    panes = []
    for _i in range(2):
        col = QVBoxLayout()
        col.setSpacing(4)
        pane_label = label("", "muted", wrap=False)
        pane_label.setAlignment(Qt.AlignCenter)
        pane_viewer = ImageViewer()
        col.addWidget(pane_label)
        col.addWidget(pane_viewer, 1)
        sl.addLayout(col, 1)
        panes.append((pane_label, pane_viewer))
    empty = label("Run a sample or open a session to review it.")
    empty.setAlignment(Qt.AlignCenter)
    views.addWidget(viewer)
    views.addWidget(side)
    views.addWidget(empty)
    views.setCurrentIndex(2)
    outer.addWidget(views, 1)

    # right: fixed-width control panel, scrolls vertically if the window is short
    scroll = QScrollArea()
    scroll.setFixedWidth(PANEL_WIDTH)
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    inner = QWidget()
    inner.setObjectName("content")
    rl = QVBoxLayout(inner)
    rl.setContentsMargins(0, 0, 6, 0)
    rl.setSpacing(12)

    conf = QFrame()
    conf.setObjectName("card")
    cl = QVBoxLayout(conf)
    cl.setContentsMargins(12, 10, 12, 10)
    cl.setSpacing(6)
    cl.addWidget(label("Confidence", "title", wrap=False))
    banner = label("", "banner")
    banner.setVisible(False)
    cl.addWidget(banner)
    badge = QLabel("")
    badge.setAlignment(Qt.AlignCenter)
    badge.setMinimumWidth(90)
    badge.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
    reason = label("")
    overall = label("", "muted")
    notes = label("", "muted")
    steps = make_details(None, "Per-step list")
    steps.text.setProperty("role", "steps")
    details = make_details(None)
    cl.addWidget(badge, 0, Qt.AlignLeft)
    cl.addWidget(reason)
    cl.addWidget(overall)
    cl.addWidget(notes)
    cl.addWidget(steps.toggle, 0, Qt.AlignLeft)
    cl.addWidget(steps.text)
    cl.addWidget(details.toggle, 0, Qt.AlignLeft)
    cl.addWidget(details.text)
    rl.addWidget(conf)

    status = label("", wrap=False)
    status.setStyleSheet("font-weight: 600;")
    rl.addWidget(status)
    export_state = label("", "muted")
    rl.addWidget(export_state)
    decide = QHBoxLayout()
    accept = button("Accept", "primary")
    reject = button("Reject")
    decide.addWidget(accept)
    decide.addWidget(reject)
    rl.addLayout(decide)

    ref_row = QHBoxLayout()
    ref_row.addWidget(label("Reference:", "muted", wrap=False))
    ref_name = label("", wrap=False)
    ref_name.setStyleSheet("font-weight: 600;")
    change_ref = button("Change", "link")
    ref_row.addWidget(ref_name)
    ref_row.addStretch(1)
    ref_row.addWidget(change_ref)
    rl.addLayout(ref_row)

    show_row = QHBoxLayout()
    show_row.addWidget(label("Show", "muted", wrap=False))
    layer = QComboBox()
    show_row.addWidget(layer, 1)
    rl.addLayout(show_row)
    onto = label("")
    rl.addWidget(onto)

    mode_buttons, mode_group, mode_row = segment_group(root, REVIEW_MODES, "blend")
    rl.addLayout(mode_row)
    slider = QSlider(Qt.Horizontal)
    slider.setRange(0, 100)
    slider.setValue(50)
    rl.addWidget(slider)
    color_buttons, color_group, color_row = segment_group(
        root, [("color", "Color"), ("gray", "Grayscale")], "color")
    color_buttons["color"].setToolTip("Reference in magenta, registered layer in green: aligned structures look "
                                      "white, misalignment shows as colored fringes.")
    rl.addLayout(color_row)
    markers = QCheckBox("Keypoints")
    markers.setChecked(True)
    caption = label("", "muted")
    rl.addWidget(markers)
    rl.addWidget(caption)

    export_accepted = button("Save accepted results...", "primary")
    export_one = button("Save this sample's results...")
    landmarks = button("Correct with landmarks (coming next)")
    landmarks.setEnabled(False)
    for b in (export_accepted, export_one, landmarks):
        rl.addWidget(b)
    rl.addStretch(1)
    scroll.setWidget(inner)
    outer.addWidget(scroll)

    return SimpleNamespace(
        widget=root, samples=samples, views=views, viewer=viewer, viewer_left=panes[0][1],
        viewer_right=panes[1][1], pane_left=panes[0][0], pane_right=panes[1][0], empty=empty, banner=banner,
        overall=overall, badge=badge, reason=reason, notes=notes, steps=steps, details=details,
        ref_name=ref_name, change_ref=change_ref, layer=layer, onto=onto, mode_buttons=mode_buttons,
        mode_group=mode_group, color_buttons=color_buttons, color_group=color_group, slider=slider,
        markers=markers, caption=caption, export_one=export_one, export_accepted=export_accepted,
        landmarks=landmarks, status=status, export_state=export_state, accept=accept, reject=reject, accept_high=accept_high,
    )

