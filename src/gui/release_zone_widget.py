"""
Crown Zone (Release/Initiation Area) Interactive Widget for PyDebFlow.

Provides an interactive matplotlib canvas embedded in PyQt6 for marking
the crown (initiation) zone on loaded terrain. Supports:
- Point mode: Click to place a circular crown zone
- Polygon mode: Click to add vertices, right-click to close
- Manual entry via Latitude / Longitude (or Row/Col for projected DEMs)
- Remove current marker
"""

import numpy as np
from typing import Optional, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSpinBox, QDoubleSpinBox,
    QButtonGroup, QRadioButton, QMessageBox
)
from PyQt6.QtCore import pyqtSignal, Qt

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class CrownWidget(QWidget):
    """
    Interactive widget for marking the crown (initiation) zone on terrain.

    Shows terrain hillshade and lets users mark point or polygon crown zones.
    Coordinate entry uses Latitude/Longitude for geographic DEMs, or
    Row/Column for projected DEMs.

    Signals:
        release_zone_changed(np.ndarray | None): Emitted when zone updates.
    """

    release_zone_changed = pyqtSignal(object)  # np.ndarray or None

    def __init__(self, parent=None):
        super().__init__(parent)

        self.terrain = None
        self.release_zone = None

        # Mode state
        self._mode = 'point'

        # Point mode
        self._point_marker = None  # (row, col)

        # Polygon mode
        self._polygon_vertices = []
        self._polygon_closed = False

        self._setup_ui()
        self._connect_signals()

    # ─────────────────────────────────────────────────────────
    # UI
    # ─────────────────────────────────────────────────────────

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(5, 2, 5, 2)

        # ── Row 1: Mode + Parameters ──────────────────────
        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        self.btn_group = QButtonGroup(self)
        self.radio_point = QRadioButton("🎯 Point")
        self.radio_point.setChecked(True)
        self.radio_point.setToolTip("Click terrain to place a circular crown zone")
        self.btn_group.addButton(self.radio_point)
        top_row.addWidget(self.radio_point)

        self.radio_polygon = QRadioButton("📐 Polygon")
        self.radio_polygon.setToolTip("Click to add vertices. Right-click to close.")
        self.btn_group.addButton(self.radio_polygon)
        top_row.addWidget(self.radio_polygon)

        top_row.addSpacing(8)

        top_row.addWidget(QLabel("H:"))
        self.height_spin = QDoubleSpinBox()
        self.height_spin.setRange(0.5, 100.0)
        self.height_spin.setValue(5.0)
        self.height_spin.setSingleStep(0.5)
        self.height_spin.setSuffix(" m")
        self.height_spin.setFixedWidth(88)
        top_row.addWidget(self.height_spin)

        top_row.addWidget(QLabel("R:"))
        self.radius_spin = QSpinBox()
        self.radius_spin.setRange(1, 50)
        self.radius_spin.setValue(10)
        self.radius_spin.setToolTip("Radius in grid cells (point mode only)")
        self.radius_spin.setFixedWidth(58)
        top_row.addWidget(self.radius_spin)

        top_row.addStretch()
        layout.addLayout(top_row)

        # ── Row 2: Coordinate Entry (Lat/Lon or Row/Col) ──
        coord_row = QHBoxLayout()
        coord_row.setSpacing(6)

        # Lat / Row label + spinbox
        self.lat_label = QLabel("Lat:")
        coord_row.addWidget(self.lat_label)
        self.lat_spin = QDoubleSpinBox()
        self.lat_spin.setDecimals(6)
        self.lat_spin.setRange(-90.0, 90.0)
        self.lat_spin.setValue(0.0)
        self.lat_spin.setFixedWidth(100)
        coord_row.addWidget(self.lat_spin)

        # Lon / Col label + spinbox
        self.lon_label = QLabel("Lon:")
        coord_row.addWidget(self.lon_label)
        self.lon_spin = QDoubleSpinBox()
        self.lon_spin.setDecimals(6)
        self.lon_spin.setRange(-180.0, 180.0)
        self.lon_spin.setValue(0.0)
        self.lon_spin.setFixedWidth(100)
        coord_row.addWidget(self.lon_spin)

        self.btn_add = QPushButton("➕ Add")
        self.btn_add.setToolTip("Point: set crown center. Polygon: add vertex.")
        self.btn_add.setFixedWidth(68)
        coord_row.addWidget(self.btn_add)

        self.btn_close_poly = QPushButton("✅ Close")
        self.btn_close_poly.setToolTip("Close polygon and compute crown zone")
        self.btn_close_poly.setEnabled(False)
        self.btn_close_poly.setFixedWidth(68)
        coord_row.addWidget(self.btn_close_poly)

        self.btn_remove = QPushButton("🗑️ Remove")
        self.btn_remove.setToolTip("Delete current crown marker")
        self.btn_remove.setFixedWidth(80)
        coord_row.addWidget(self.btn_remove)

        coord_row.addStretch()

        self.status_label = QLabel("No terrain loaded")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        coord_row.addWidget(self.status_label)

        layout.addLayout(coord_row)

        # ── Matplotlib Canvas ──────────────────────────────
        self.figure = Figure(figsize=(6, 5), dpi=100, facecolor='#16213e')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setMinimumHeight(350)
        self.ax = self.figure.add_subplot(111)
        self._init_ax()
        self.figure.tight_layout(pad=0.5)
        layout.addWidget(self.canvas, stretch=1)

    def _init_ax(self):
        self.ax.set_facecolor('#1a1a2e')
        self.ax.set_title("Load terrain to begin", color='#e8e8e8', fontsize=10)
        self._style_ax(self.ax)

    def _connect_signals(self):
        self.radio_point.toggled.connect(self._on_mode_changed)
        self.radio_polygon.toggled.connect(self._on_mode_changed)
        self.btn_add.clicked.connect(self._on_manual_add)
        self.btn_close_poly.clicked.connect(self._on_close_polygon)
        self.btn_remove.clicked.connect(self.clear_zone)
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.canvas.mpl_connect('scroll_event', self._on_scroll)

    # ─────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────

    def set_terrain(self, terrain):
        self.terrain = terrain
        self.release_zone = None
        self._point_marker = None
        self._polygon_vertices = []
        self._polygon_closed = False

        # Update coordinate spinbox ranges / labels
        if terrain.is_geographic:
            self.lat_label.setText("Lat:")
            self.lon_label.setText("Lon:")
            # lat/lon bounds: y_origin = bottom, x_origin = left
            lat_min = terrain.y_origin
            lat_max = terrain.y_origin + terrain.rows * terrain.cell_size_deg
            lon_min = terrain.x_origin
            lon_max = terrain.x_origin + terrain.cols * terrain.cell_size_deg
            self.lat_spin.setRange(lat_min, lat_max)
            self.lon_spin.setRange(lon_min, lon_max)
            # Default to centre
            self.lat_spin.setValue((lat_min + lat_max) / 2)
            self.lon_spin.setValue((lon_min + lon_max) / 2)
            self.lat_spin.setDecimals(6)
            self.lon_spin.setDecimals(6)
        else:
            self.lat_label.setText("Row:")
            self.lon_label.setText("Col:")
            self.lat_spin.setRange(0, terrain.rows - 1)
            self.lon_spin.setRange(0, terrain.cols - 1)
            self.lat_spin.setValue(terrain.rows // 5)
            self.lon_spin.setValue(terrain.cols // 2)
            self.lat_spin.setDecimals(0)
            self.lon_spin.setDecimals(0)

        self._draw_terrain()
        self.status_label.setText("Click terrain to mark crown zone")

    def get_release_zone(self) -> Optional[np.ndarray]:
        return self.release_zone

    def clear_zone(self):
        """Remove current crown marker."""
        self.release_zone = None
        self._point_marker = None
        self._polygon_vertices = []
        self._polygon_closed = False
        self.btn_close_poly.setEnabled(False)
        if self.terrain is not None:
            self._draw_terrain()
            self.status_label.setText("Cleared — click to mark new zone")
        self.release_zone_changed.emit(None)

    # ─────────────────────────────────────────────────────────
    # Coord conversion helpers
    # ─────────────────────────────────────────────────────────

    def _latlon_to_rowcol(self, lat: float, lon: float) -> Tuple[int, int]:
        """Convert geographic lat/lon to grid row/col."""
        t = self.terrain
        if t.is_geographic and t.cell_size_deg > 0:
            # row 0 = top of raster (max lat), row increases downward
            lat_top = t.y_origin + t.rows * t.cell_size_deg
            row = int((lat_top - lat) / t.cell_size_deg)
            col = int((lon - t.x_origin) / t.cell_size_deg)
        else:
            row = int(round(lat))  # lat_spin used as row
            col = int(round(lon))
        row = max(0, min(row, t.rows - 1))
        col = max(0, min(col, t.cols - 1))
        return row, col

    def _rowcol_to_latlon(self, row: int, col: int) -> Tuple[float, float]:
        """Convert grid row/col to geographic lat/lon (or row/col if projected)."""
        t = self.terrain
        if t.is_geographic and t.cell_size_deg > 0:
            lat_top = t.y_origin + t.rows * t.cell_size_deg
            lat = lat_top - row * t.cell_size_deg
            lon = t.x_origin + col * t.cell_size_deg
        else:
            lat, lon = float(row), float(col)
        return lat, lon

    def _update_coord_spinboxes(self, row: int, col: int):
        lat, lon = self._rowcol_to_latlon(row, col)
        self.lat_spin.blockSignals(True)
        self.lon_spin.blockSignals(True)
        self.lat_spin.setValue(lat)
        self.lon_spin.setValue(lon)
        self.lat_spin.blockSignals(False)
        self.lon_spin.blockSignals(False)

    # ─────────────────────────────────────────────────────────
    # Drawing
    # ─────────────────────────────────────────────────────────

    def _draw_terrain(self):
        self.ax.clear()
        if self.terrain is None:
            self.ax.set_title("No terrain loaded", color='#e8e8e8', fontsize=10)
            self.canvas.draw_idle()
            return

        hillshade = self.terrain.get_hillshade()
        self.ax.imshow(hillshade, cmap='gray', origin='upper', aspect='equal', interpolation='bilinear')
        self.ax.contour(self.terrain.elevation, levels=15, colors='#00d9ff', linewidths=0.3, alpha=0.4)

        self.ax.set_title("Crown Zone — Click to mark initiation area", color='#e8e8e8', fontsize=10)
        self.ax.set_xlabel("Column (j)", color='#888', fontsize=8)
        self.ax.set_ylabel("Row (i)", color='#888', fontsize=8)
        self._style_ax(self.ax)

        self._draw_overlays()
        self.figure.tight_layout(pad=0.5)
        self.canvas.draw_idle()

    def _draw_overlays(self):
        # Release zone heatmap
        if self.release_zone is not None:
            masked = np.ma.masked_where(self.release_zone < 0.01, self.release_zone)
            self.ax.imshow(masked, cmap='hot', alpha=0.55, origin='upper', aspect='equal', interpolation='bilinear')

        # Point marker
        if self._point_marker is not None:
            r, c = self._point_marker
            from matplotlib.patches import Circle
            circle = Circle((c, r), self.radius_spin.value(),
                            fill=False, edgecolor='#00ff88', linewidth=2, linestyle='--')
            self.ax.add_patch(circle)
            self.ax.plot(c, r, 'x', color='#00ff88', markersize=10, markeredgewidth=2)

        # Polygon vertices/edges
        if self._polygon_vertices:
            verts = self._polygon_vertices
            rows = [v[0] for v in verts]
            cols = [v[1] for v in verts]
            self.ax.plot(cols, rows, 'o', color='#ff6b6b', markersize=6,
                         markeredgecolor='white', markeredgewidth=1)
            if len(verts) > 1:
                pc = cols + ([cols[0]] if self._polygon_closed else [])
                pr = rows + ([rows[0]] if self._polygon_closed else [])
                self.ax.plot(pc, pr, '-', color='#ff6b6b', linewidth=1.5)
            if self._polygon_closed and len(verts) >= 3:
                from matplotlib.patches import Polygon as MplPoly
                xy = np.array([[c, r] for r, c in verts])
                self.ax.add_patch(MplPoly(xy, closed=True, facecolor='#ff6b6b', alpha=0.2,
                                          edgecolor='#ff6b6b', linewidth=2))

    # ─────────────────────────────────────────────────────────
    # Event Handlers
    # ─────────────────────────────────────────────────────────

    def _on_mode_changed(self, checked):
        if not checked:
            return
        old = self._mode
        self._mode = 'point' if self.radio_point.isChecked() else 'polygon'
        if old != self._mode:
            self._point_marker = None
            self._polygon_vertices = []
            self._polygon_closed = False
            self.btn_close_poly.setEnabled(False)
            self.radius_spin.setEnabled(self._mode == 'point')
            if self.terrain is not None:
                self._draw_terrain()

    def _on_canvas_click(self, event):
        if self.terrain is None or event.inaxes != self.ax:
            return
        col = int(round(event.xdata))
        row = int(round(event.ydata))
        row = max(0, min(row, self.terrain.rows - 1))
        col = max(0, min(col, self.terrain.cols - 1))

        if self._mode == 'point':
            self._place_point(row, col)
        else:
            if event.button == 3:
                self._close_polygon()
            else:
                self._add_vertex(row, col)

    def _on_scroll(self, event):
        """Scroll-wheel zoom on the terrain canvas."""
        if event.inaxes != self.ax:
            return
        factor = 0.85 if event.button == 'up' else 1.15
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xm, ym = event.xdata, event.ydata
        self.ax.set_xlim([xm + (x - xm) * factor for x in xlim])
        self.ax.set_ylim([ym + (y - ym) * factor for y in ylim])
        self.canvas.draw_idle()

    def _on_manual_add(self):
        if self.terrain is None:
            QMessageBox.warning(self, "No Terrain", "Load a DEM first.")
            return
        row, col = self._latlon_to_rowcol(self.lat_spin.value(), self.lon_spin.value())
        if self._mode == 'point':
            self._place_point(row, col)
        else:
            self._add_vertex(row, col)

    def _on_close_polygon(self):
        self._close_polygon()

    # ─────────────────────────────────────────────────────────
    # Crown Zone Logic
    # ─────────────────────────────────────────────────────────

    def _place_point(self, row: int, col: int):
        self._point_marker = (row, col)
        height = self.height_spin.value()
        radius = self.radius_spin.value()
        self.release_zone = self.terrain.create_release_zone(
            center_i=row, center_j=col, radius=radius, height=height
        )
        self._update_coord_spinboxes(row, col)
        lat, lon = self._rowcol_to_latlon(row, col)
        if self.terrain.is_geographic:
            self.status_label.setText(f"Crown: ({lat:.5f}°, {lon:.5f}°) r={radius}")
        else:
            self.status_label.setText(f"Crown: ({row}, {col}) r={radius}")
        self._draw_terrain()
        self.release_zone_changed.emit(self.release_zone)

    def _add_vertex(self, row: int, col: int):
        if self._polygon_closed:
            self._polygon_vertices = []
            self._polygon_closed = False
            self.release_zone = None
        self._polygon_vertices.append((row, col))
        n = len(self._polygon_vertices)
        self.btn_close_poly.setEnabled(n >= 3)
        need = max(0, 3 - n)
        self.status_label.setText(
            f"Polygon: {n} vertices" + (f" — right-click or Close" if n >= 3 else f" — need {need} more")
        )
        self._draw_terrain()

    def _close_polygon(self):
        if len(self._polygon_vertices) < 3:
            QMessageBox.warning(self, "Not Enough Vertices", "A polygon needs at least 3 vertices.")
            return
        self._polygon_closed = True
        height = self.height_spin.value()
        self.release_zone = self.terrain.create_polygon_release_zone(
            vertices=self._polygon_vertices, height=height, smooth=True
        )
        n = len(self._polygon_vertices)
        self.status_label.setText(f"Polygon ({n} vertices) h={height:.1f}m ✓")
        self.btn_close_poly.setEnabled(False)
        self._draw_terrain()
        self.release_zone_changed.emit(self.release_zone)

    # ─────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────

    def _style_ax(self, ax):
        ax.tick_params(colors='#888888', labelsize=7)
        for spine in ax.spines.values():
            spine.set_color('#3d3d5c')


# Backward-compatibility alias
ReleaseZoneWidget = CrownWidget
