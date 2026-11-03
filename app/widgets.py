"""The classes in the app: Qt requires subclasses for these four."""
from __future__ import annotations

import threading
from typing import List, Optional

import numpy as np
import pandas as pd
from PyQt5.QtCore import QPointF, QRectF, Qt, QThread, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap
from PyQt5.QtWidgets import (
    QFrame, QGraphicsEllipseItem, QGraphicsItem, QGraphicsScene, QGraphicsView,
)

import pipeline

MARKER_RADIUS_PX = 2.5


def array_to_qimage(arr: np.ndarray) -> QImage:
    """uint8 HxWx3 -> QImage (copied, so the array can be freed)."""
    arr = np.ascontiguousarray(arr, dtype=np.uint8)
    h, w = arr.shape[:2]
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


class DropFrame(QFrame):
    """A frame that accepts dropped files and folders and reports them.
    While a drag is over it, its "state" property becomes "over" (the
    stylesheet highlights it); it returns to the previous state after."""

    paths_dropped = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self._state_before = None

    def _set_state(self, state):
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            self._state_before = self.property("state")
            self._set_state("over")
            event.acceptProposedAction()

    def dragLeaveEvent(self, event):
        self._set_state(self._state_before)

    def dropEvent(self, event):
        self._set_state(self._state_before)
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.paths_dropped.emit(paths)
            event.acceptProposedAction()



class ImageViewer(QGraphicsView):
    """Shows one RGB uint8 array. Wheel zooms around the cursor, left-drag
    pans. Keypoint markers are screen-size circles that can be toggled."""

    ZOOM_FACTOR = 1.15
    MAX_ZOOM_OVER_FIT = 60.0
    MIN_ZOOM_OVER_FIT = 0.5

    view_changed = pyqtSignal()
    cursor_moved = pyqtSignal(object)  # QPointF in image coordinates, or None
    CURSOR_RADIUS_PX = 7.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pix_item = None
        self._shape = None
        self._markers: List[QGraphicsEllipseItem] = []
        self._cursor_item = None
        self._auto_fit = True
        self._scale = 1.0
        self._center = QPointF(0.0, 0.0)
        self._drag_from = None
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setTransformationAnchor(QGraphicsView.NoAnchor)
        self.setResizeAnchor(QGraphicsView.NoAnchor)
        self.setFrameShape(QFrame.NoFrame)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setBackgroundBrush(QBrush(QColor("#111111")))
        self.setMinimumSize(200, 160)

    def set_image(self, rgb: np.ndarray) -> None:
        """Show rgb; the view (zoom and pan) is kept when the new image has
        the same shape as the current one, otherwise it is fitted."""
        pixmap = QPixmap.fromImage(array_to_qimage(rgb))
        same = self._shape == rgb.shape
        if self._pix_item is None:
            self._pix_item = self._scene.addPixmap(pixmap)
        else:
            self._pix_item.setPixmap(pixmap)
        w, h = pixmap.width(), pixmap.height()
        # padded scene so zooming out and panning past the edge are never clamped
        self._scene.setSceneRect(QRectF(-w, -h, 3 * w, 3 * h))
        self._shape = rgb.shape
        if not same:
            self._auto_fit = True
            self.fit_view()

    def clear(self) -> None:
        self._scene.clear()
        self._pix_item = None
        self._cursor_item = None
        self._shape = None
        self._markers = []

    def _fit_scale(self) -> float:
        if self._pix_item is None:
            return 1.0
        rect = self._pix_item.pixmap().rect()
        vp = self.viewport().rect()
        return max(min(vp.width() / max(rect.width(), 1), vp.height() / max(rect.height(), 1)), 1e-6)

    def _apply(self, scale: float, center: QPointF, emit: bool = True) -> None:
        self._scale, self._center = scale, QPointF(center)
        self.resetTransform()
        self.scale(scale, scale)
        self.centerOn(center)
        if emit:
            self.view_changed.emit()

    def fit_view(self, emit: bool = True) -> None:
        if self._pix_item is None:
            return
        rect = self._pix_item.pixmap().rect()
        self._auto_fit = True
        self._apply(self._fit_scale(), QPointF(rect.width() / 2.0, rect.height() / 2.0), emit)

    def view_state(self):
        """(scale, center x, center y, auto_fit): enough to reproduce the view."""
        return self._scale, self._center.x(), self._center.y(), self._auto_fit

    def set_view_state(self, state) -> None:
        """Apply a view_state() from another viewer without emitting view_changed."""
        scale, cx, cy, auto = state
        self._auto_fit = bool(auto)
        self._apply(scale, QPointF(cx, cy), emit=False)

    def visible_rect(self) -> QRectF:
        """The part of the scene (image pixels) shown in the viewport."""
        return self.mapToScene(self.viewport().rect()).boundingRect()

    def set_cursor_marker(self, point: Optional[QPointF]) -> None:
        """A small ring at an image coordinate (None hides it); it keeps its
        on-screen size at any zoom."""
        if point is None or self._pix_item is None:
            if self._cursor_item is not None:
                self._cursor_item.setVisible(False)
            return
        if self._cursor_item is None or self._cursor_item.scene() is None:
            r = self.CURSOR_RADIUS_PX
            self._cursor_item = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
            self._cursor_item.setPen(QPen(QColor("#00e5ff"), 1.5))
            self._cursor_item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
            self._cursor_item.setZValue(20)
            self._scene.addItem(self._cursor_item)
        self._cursor_item.setPos(point)
        self._cursor_item.setVisible(True)

    def set_markers(self, points: np.ndarray, offsets_x: Optional[List[float]] = None) -> None:
        """Place a marker per (x, y) point, repeated at each x offset (the
        side-by-side mode shows the same points on both panels)."""
        for item in self._markers:
            self._scene.removeItem(item)
        self._markers = []
        for dx in (offsets_x or [0.0]):
            for x, y in np.asarray(points, dtype=float).reshape(-1, 2):
                r = MARKER_RADIUS_PX
                item = QGraphicsEllipseItem(-r, -r, 2 * r, 2 * r)
                item.setPen(QPen(QColor("#ffd400"), 1.0))
                item.setBrush(QBrush(Qt.NoBrush))
                item.setFlag(QGraphicsItem.ItemIgnoresTransformations, True)
                item.setPos(QPointF(x + dx, y))
                item.setZValue(10)
                self._scene.addItem(item)
                self._markers.append(item)

    def set_markers_visible(self, visible: bool) -> None:
        for item in self._markers:
            item.setVisible(visible)

    def marker_count(self) -> int:
        return len(self._markers)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._auto_fit:
            self.fit_view()
        else:
            self._apply(self._scale, self._center)

    def wheelEvent(self, event):
        if self._pix_item is None:
            return
        factor = self.ZOOM_FACTOR if event.angleDelta().y() > 0 else 1.0 / self.ZOOM_FACTOR
        fit = self._fit_scale()
        new_scale = min(max(self._scale * factor, fit * self.MIN_ZOOM_OVER_FIT), fit * self.MAX_ZOOM_OVER_FIT)
        pos = event.pos()
        vp = self.viewport().rect()
        # keep the scene point under the cursor at the same viewport position
        under = QPointF(self._center.x() + (pos.x() - vp.width() / 2.0) / self._scale,
                        self._center.y() + (pos.y() - vp.height() / 2.0) / self._scale)
        center = QPointF(under.x() - (pos.x() - vp.width() / 2.0) / new_scale,
                         under.y() - (pos.y() - vp.height() / 2.0) / new_scale)
        self._auto_fit = False
        self._apply(new_scale, center)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_from = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event):
        if self._drag_from is not None and self._pix_item is not None:
            delta = event.pos() - self._drag_from
            self._drag_from = event.pos()
            self._auto_fit = False
            self._apply(self._scale, QPointF(self._center.x() - delta.x() / self._scale,
                                             self._center.y() - delta.y() / self._scale))
        self.cursor_moved.emit(self.mapToScene(event.pos()) if self._pix_item is not None else None)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._drag_from = None
            self.unsetCursor()
        event.accept()

    def mouseDoubleClickEvent(self, event):
        self.fit_view()
        event.accept()

    def leaveEvent(self, event):
        self.cursor_moved.emit(None)
        super().leaveEvent(event)

    def zoom_scale(self) -> float:
        return self._scale


class BatchThread(QThread):
    """Runs pipeline.run_batch off the UI thread. cancel_event is checked by
    run_batch and run_sample between registration steps."""

    progress = pyqtSignal(float, str)
    sample_done = pyqtSignal(object)  # one result, emitted as soon as that sample finishes
    batch_done = pyqtSignal(object, object)  # results list, summary table
    batch_error = pyqtSignal(str)

    def __init__(self, samples: List[dict], target_um: float, parent=None):
        super().__init__(parent)
        self._samples = samples
        self._target_um = float(target_um)
        self.cancel_event = threading.Event()

    def run(self):
        previous = pipeline.WORKING_UM_PER_PX
        pipeline.WORKING_UM_PER_PX = self._target_um
        try:
            results, tables = [], []
            for sample in self._samples:
                # one sample per run_batch call, so each result is available (and autosaved) when it finishes
                one, table = pipeline.run_batch(
                    [sample], progress=lambda f, m: self.progress.emit(float(f), str(m)), cancel=self.cancel_event
                )
                results += one
                tables.append(table)
                self.sample_done.emit(one[0])
            self.batch_done.emit(results, pd.concat(tables, ignore_index=True) if tables else None)
        except Exception as exc:  # noqa: BLE001 - reported in the UI
            self.batch_error.emit(f"{type(exc).__name__}: {exc}")
        finally:
            pipeline.WORKING_UM_PER_PX = previous
