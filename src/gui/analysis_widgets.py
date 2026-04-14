"""
Analysis Widgets for PyDebFlow v0.2.0.

RAMMS-style post-simulation analysis tools:
- CrossSectionWidget  : draw a transect, view elevation + flow profile, zoom/save
- HydrographWidget    : multi-point hydrograph (height, velocity, discharge), save
- StatisticsWidget    : descriptive, spatial, temporal statistics + save CSV/TXT
"""

import numpy as np
from typing import Optional, List, Tuple
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QSlider, QPushButton, QSpinBox, QDoubleSpinBox,
    QSplitter, QFileDialog, QTextEdit, QScrollArea,
    QGroupBox, QGridLayout, QMessageBox
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt import NavigationToolbar2QT as NavToolbar
from matplotlib.figure import Figure

# ─────────────────────────────────────────────────────────────────────────────
# Colour cycle for multi-point overlays
# ─────────────────────────────────────────────────────────────────────────────
_COLOURS = ['#ff4444', '#44aaff', '#ffcc00', '#44ff88',
            '#ff88cc', '#88ffcc', '#cc88ff', '#ff8844']


# ─────────────────────────────────────────────────────────────────────────────
# CrossSectionWidget
# ─────────────────────────────────────────────────────────────────────────────

class CrossSectionWidget(QWidget):
    """
    RAMMS-style cross-section profile viewer.

    • Click two points on the terrain map for a transect.
    • Profile x-axis sampled every 0.1 m (capped at 5000 pts).
    • Time slider to scrub through timesteps.
    • Zoom/Pan toolbar + Save Plot button.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.terrain = None
        self.outputs = None
        self._pt1 = None
        self._pt2 = None
        self._current_frame = 0
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 2, 4, 2)

        # Info bar
        self.info_label = QLabel("Click two points on the terrain to define a cross-section line")
        self.info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self.info_label)

        # Splitter: terrain map + profile
        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Terrain map ───────────────────────────────────
        self.map_figure = Figure(figsize=(5, 3), dpi=100, facecolor='#16213e')
        self.map_canvas = FigureCanvas(self.map_figure)
        self.map_ax = self.map_figure.add_subplot(111)
        self.map_ax.set_facecolor('#1a1a2e')
        self._style_ax(self.map_ax)
        self.map_figure.tight_layout(pad=0.5)
        self.map_canvas.mpl_connect('button_press_event', self._on_map_click)
        self.map_canvas.mpl_connect('scroll_event', self._on_map_scroll)
        splitter.addWidget(self.map_canvas)

        # ── Profile plot ──────────────────────────────────
        profile_container = QWidget()
        pc_layout = QVBoxLayout(profile_container)
        pc_layout.setContentsMargins(0, 0, 0, 0)
        pc_layout.setSpacing(2)

        self.profile_figure = Figure(figsize=(5, 2.5), dpi=100, facecolor='#16213e')
        self.profile_canvas = FigureCanvas(self.profile_figure)
        self.profile_ax = self.profile_figure.add_subplot(111)
        self.profile_ax.set_facecolor('#1a1a2e')
        self._style_ax(self.profile_ax)
        self.profile_figure.tight_layout(pad=0.5)
        pc_layout.addWidget(self.profile_canvas, stretch=1)

        # Toolbar + Save
        tb_row = QHBoxLayout()
        nav = NavToolbar(self.profile_canvas, profile_container)
        nav.setStyleSheet("background: #1a1a2e; color: #ccc;")
        tb_row.addWidget(nav)
        btn_save = QPushButton("💾 Save Plot")
        btn_save.setFixedWidth(90)
        btn_save.clicked.connect(self._save_profile)
        tb_row.addWidget(btn_save)
        pc_layout.addLayout(tb_row)

        splitter.addWidget(profile_container)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, stretch=1)

        # Time slider row
        slider_row = QHBoxLayout()
        slider_row.addWidget(QLabel("Time:"))
        self.time_slider = QSlider(Qt.Orientation.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(0)
        self.time_slider.valueChanged.connect(self._on_slider)
        slider_row.addWidget(self.time_slider, stretch=1)
        self.time_label = QLabel("t = 0.0 s")
        self.time_label.setFixedWidth(90)
        slider_row.addWidget(self.time_label)
        btn_reset = QPushButton("🔄 Reset")
        btn_reset.setFixedWidth(72)
        btn_reset.clicked.connect(self._reset_line)
        slider_row.addWidget(btn_reset)
        layout.addLayout(slider_row)

    def set_data(self, terrain, outputs):
        self.terrain = terrain
        self.outputs = outputs
        self._pt1 = None
        self._pt2 = None
        self._current_frame = 0
        if outputs:
            self.time_slider.setMaximum(len(outputs) - 1)
            self.time_slider.setValue(0)
        self._draw_map()

    # ── Drawing ───────────────────────────────────────────

    def _draw_map(self):
        self.map_ax.clear()
        if self.terrain is None:
            self.map_canvas.draw_idle()
            return
        hs = self.terrain.get_hillshade()
        self.map_ax.imshow(hs, cmap='gray', origin='upper', aspect='equal')
        if self.outputs:
            _, state = self.outputs[self._current_frame]
            h = state.h_solid + state.h_fluid
            masked = np.ma.masked_where(h < 0.01, h)
            self.map_ax.imshow(masked, cmap='YlOrRd', alpha=0.5, origin='upper', aspect='equal')
        for pt in [self._pt1, self._pt2]:
            if pt is not None:
                self.map_ax.plot(pt[1], pt[0], 'o', color='#00ff88', markersize=8,
                                 markeredgecolor='white', markeredgewidth=1.5)
        if self._pt1 is not None and self._pt2 is not None:
            self.map_ax.plot([self._pt1[1], self._pt2[1]], [self._pt1[0], self._pt2[0]],
                             '--', color='#00ff88', linewidth=2)
        self.map_ax.set_title("Terrain — click two points for cross-section", color='#e8e8e8', fontsize=9)
        self._style_ax(self.map_ax)
        self.map_figure.tight_layout(pad=0.5)
        self.map_canvas.draw_idle()

    def _draw_profile(self):
        self.profile_ax.clear()
        if self._pt1 is None or self._pt2 is None or self.terrain is None:
            self.profile_ax.set_title("Select two points on the map above", color='#e8e8e8', fontsize=9)
            self._style_ax(self.profile_ax)
            self.profile_figure.tight_layout(pad=0.5)
            self.profile_canvas.draw_idle()
            return

        r1, c1 = self._pt1
        r2, c2 = self._pt2
        cs = self.terrain.cell_size
        total_dist = np.sqrt(((r2 - r1) * cs) ** 2 + ((c2 - c1) * cs) ** 2)

        # x-axis at 0.1 m resolution, capped at 5000 samples
        n_samples = max(2, min(int(total_dist / 0.1), 5000))
        rows_l = np.linspace(r1, r2, n_samples)
        cols_l = np.linspace(c1, c2, n_samples)
        dist = np.linspace(0, total_dist, n_samples)

        ri = np.clip(np.round(rows_l).astype(int), 0, self.terrain.rows - 1)
        ci = np.clip(np.round(cols_l).astype(int), 0, self.terrain.cols - 1)
        elev = self.terrain.elevation[ri, ci]

        self.profile_ax.fill_between(dist, elev.min() - 5, elev, color='#8B7355', alpha=0.35)
        self.profile_ax.plot(dist, elev, '-', color='#8B7355', linewidth=2, label='Terrain')

        if self.outputs:
            t, state = self.outputs[self._current_frame]
            h_total = (state.h_solid + state.h_fluid)[ri, ci]
            flow_surf = elev + h_total
            mask = h_total > 0.01
            if mask.any():
                self.profile_ax.fill_between(dist, elev, flow_surf, where=mask,
                                              color='#ff4444', alpha=0.45)
                self.profile_ax.plot(dist[mask], flow_surf[mask], '-',
                                     color='#ff4444', linewidth=1.5, label=f'Flow (t={t:.1f}s)')

        self.profile_ax.set_xlabel("Distance (m)", color='#aaa', fontsize=8)
        self.profile_ax.set_ylabel("Elevation (m)", color='#aaa', fontsize=8)
        self.profile_ax.set_title(f"Cross-Section Profile  [{total_dist:.0f} m transect, ~{n_samples} pts]",
                                   color='#e8e8e8', fontsize=9)
        self.profile_ax.legend(loc='upper right', fontsize=7,
                                facecolor='#2a2a4a', edgecolor='#555', labelcolor='#ddd')
        self._style_ax(self.profile_ax)
        self.profile_figure.tight_layout(pad=0.5)
        self.profile_canvas.draw_idle()

    # ── Events ────────────────────────────────────────────

    def _on_map_click(self, event):
        if self.terrain is None or event.inaxes != self.map_ax:
            return
        col = max(0, min(int(round(event.xdata)), self.terrain.cols - 1))
        row = max(0, min(int(round(event.ydata)), self.terrain.rows - 1))
        if self._pt1 is None:
            self._pt1 = (row, col)
            self.info_label.setText(f"Point A: ({row}, {col}) — click second point")
        elif self._pt2 is None:
            self._pt2 = (row, col)
            cs = self.terrain.cell_size
            length = np.sqrt(((self._pt2[0]-self._pt1[0])*cs)**2 + ((self._pt2[1]-self._pt1[1])*cs)**2)
            self.info_label.setText(f"A({self._pt1[0]},{self._pt1[1]}) → B({self._pt2[0]},{self._pt2[1]}) | {length:.0f} m")
        else:
            self._pt1 = (row, col); self._pt2 = None
            self.info_label.setText(f"Point A: ({row}, {col}) — click second point")
        self._draw_map(); self._draw_profile()

    def _on_map_scroll(self, event):
        if event.inaxes != self.map_ax:
            return
        f = 0.85 if event.button == 'up' else 1.15
        xl, yl = self.map_ax.get_xlim(), self.map_ax.get_ylim()
        xm, ym = event.xdata, event.ydata
        self.map_ax.set_xlim([xm + (x - xm)*f for x in xl])
        self.map_ax.set_ylim([ym + (y - ym)*f for y in yl])
        self.map_canvas.draw_idle()

    def _on_slider(self, value):
        self._current_frame = value
        if self.outputs and value < len(self.outputs):
            self.time_label.setText(f"t = {self.outputs[value][0]:.1f} s")
        self._draw_map(); self._draw_profile()

    def _reset_line(self):
        self._pt1 = None; self._pt2 = None
        self.info_label.setText("Click two points on the terrain to define a cross-section line")
        self._draw_map(); self._draw_profile()

    def _save_profile(self):
        path, fmt = QFileDialog.getSaveFileName(
            self, "Save Profile Plot", "profile.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self.profile_figure.savefig(path, dpi=150, bbox_inches='tight',
                                         facecolor=self.profile_figure.get_facecolor())

    def _style_ax(self, ax):
        ax.tick_params(colors='#888', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#3d3d5c')


# ─────────────────────────────────────────────────────────────────────────────
# HydrographWidget
# ─────────────────────────────────────────────────────────────────────────────

class HydrographWidget(QWidget):
    """
    Multi-point hydrograph viewer.

    • Click or type coordinates to add monitor points.
    • All points plotted on the same graph with distinct colours.
    • Zoom/Pan toolbar + Save Plot button.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.terrain = None
        self.outputs = None
        self._monitor_points: List[Tuple[int, int]] = []
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 2, 4, 2)

        # Info
        self.info_label = QLabel("Click a point on the terrain or enter coordinates below")
        self.info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        layout.addWidget(self.info_label)

        # Coordinate entry row
        entry = QHBoxLayout()
        entry.addWidget(QLabel("Row:"))
        self.row_spin = QSpinBox(); self.row_spin.setRange(0, 9999); self.row_spin.setFixedWidth(70)
        entry.addWidget(self.row_spin)
        entry.addWidget(QLabel("Col:"))
        self.col_spin = QSpinBox(); self.col_spin.setRange(0, 9999); self.col_spin.setFixedWidth(70)
        entry.addWidget(self.col_spin)

        self.btn_set = QPushButton("📍 Set Point")
        self.btn_set.setFixedWidth(90)
        self.btn_set.setToolTip("Replace all points with this one")
        self.btn_set.clicked.connect(self._on_set_point)
        entry.addWidget(self.btn_set)

        self.btn_add = QPushButton("➕ Add to Graph")
        self.btn_add.setFixedWidth(105)
        self.btn_add.setToolTip("Add this point to the current graph (keep others)")
        self.btn_add.clicked.connect(self._on_add_point)
        entry.addWidget(self.btn_add)

        self.btn_clear = QPushButton("🗑️ Clear All")
        self.btn_clear.setFixedWidth(85)
        self.btn_clear.clicked.connect(self._clear_all_points)
        entry.addWidget(self.btn_clear)

        entry.addStretch()
        layout.addLayout(entry)

        # Splitter: terrain + hydrograph
        splitter = QSplitter(Qt.Orientation.Vertical)

        # ── Terrain map ───────────────────────────────────
        self.map_figure = Figure(figsize=(5, 2.5), dpi=100, facecolor='#16213e')
        self.map_canvas = FigureCanvas(self.map_figure)
        self.map_ax = self.map_figure.add_subplot(111)
        self.map_ax.set_facecolor('#1a1a2e')
        self._style_ax(self.map_ax)
        self.map_figure.tight_layout(pad=0.5)
        self.map_canvas.mpl_connect('button_press_event', self._on_map_click)
        self.map_canvas.mpl_connect('scroll_event', self._on_map_scroll)
        splitter.addWidget(self.map_canvas)

        # ── Hydrograph plots ──────────────────────────────
        hydro_container = QWidget()
        hc_layout = QVBoxLayout(hydro_container)
        hc_layout.setContentsMargins(0, 0, 0, 0)
        hc_layout.setSpacing(2)

        self.hydro_figure = Figure(figsize=(5, 3.5), dpi=100, facecolor='#16213e')
        self.hydro_canvas = FigureCanvas(self.hydro_figure)
        self.ax_height = self.hydro_figure.add_subplot(211)
        self.ax_vel = self.hydro_figure.add_subplot(212)
        for ax in [self.ax_height, self.ax_vel]:
            ax.set_facecolor('#1a1a2e')
            self._style_ax(ax)
        self.hydro_figure.tight_layout(pad=0.8)
        hc_layout.addWidget(self.hydro_canvas, stretch=1)

        # Toolbar + Save
        tb_row2 = QHBoxLayout()
        nav2 = NavToolbar(self.hydro_canvas, hydro_container)
        nav2.setStyleSheet("background: #1a1a2e; color: #ccc;")
        tb_row2.addWidget(nav2)
        btn_save2 = QPushButton("💾 Save Plot")
        btn_save2.setFixedWidth(90)
        btn_save2.clicked.connect(self._save_hydro)
        tb_row2.addWidget(btn_save2)
        hc_layout.addLayout(tb_row2)

        splitter.addWidget(hydro_container)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 3)
        layout.addWidget(splitter, stretch=1)

    def set_data(self, terrain, outputs):
        self.terrain = terrain
        self.outputs = outputs
        self._monitor_points = []
        if terrain:
            self.row_spin.setRange(0, terrain.rows - 1)
            self.col_spin.setRange(0, terrain.cols - 1)
            self.row_spin.setValue(terrain.rows // 2)
            self.col_spin.setValue(terrain.cols // 2)
        self._draw_map()
        self._draw_hydrograph()

    # ── Drawing ───────────────────────────────────────────

    def _draw_map(self):
        self.map_ax.clear()
        if self.terrain is None:
            self.map_canvas.draw_idle()
            return
        hs = self.terrain.get_hillshade()
        self.map_ax.imshow(hs, cmap='gray', origin='upper', aspect='equal')
        if self.outputs:
            max_h = np.zeros((self.terrain.rows, self.terrain.cols))
            for _, s in self.outputs:
                max_h = np.maximum(max_h, s.h_solid + s.h_fluid)
            masked = np.ma.masked_where(max_h < 0.01, max_h)
            self.map_ax.imshow(masked, cmap='YlOrRd', alpha=0.4, origin='upper', aspect='equal')
        for idx, (r, c) in enumerate(self._monitor_points):
            colour = _COLOURS[idx % len(_COLOURS)]
            self.map_ax.plot(c, r, 's', color=colour, markersize=9,
                             markeredgecolor='white', markeredgewidth=1.5)
            self.map_ax.annotate(str(idx + 1), (c, r), color='white', fontsize=7,
                                 ha='center', va='center', fontweight='bold')
        self.map_ax.set_title("Click to add monitor point", color='#e8e8e8', fontsize=9)
        self._style_ax(self.map_ax)
        self.map_figure.tight_layout(pad=0.5)
        self.map_canvas.draw_idle()

    def _draw_hydrograph(self):
        self.ax_height.clear()
        self.ax_vel.clear()
        if not self._monitor_points or not self.outputs:
            self.ax_height.set_title("Select one or more monitor points", color='#e8e8e8', fontsize=9)
            for ax in [self.ax_height, self.ax_vel]:
                self._style_ax(ax)
            self.hydro_figure.tight_layout(pad=0.8)
            self.hydro_canvas.draw_idle()
            return

        times = np.array([t for t, _ in self.outputs])

        for idx, (r, c) in enumerate(self._monitor_points):
            colour = _COLOURS[idx % len(_COLOURS)]
            label = f"Pt{idx+1} ({r},{c})"

            h_s = np.array([st.h_solid[r, c] for _, st in self.outputs])
            h_f = np.array([st.h_fluid[r, c] for _, st in self.outputs])
            h_tot = h_s + h_f
            vs = np.array([np.sqrt(st.u_solid[r,c]**2 + st.v_solid[r,c]**2) for _, st in self.outputs])
            vf = np.array([np.sqrt(st.u_fluid[r,c]**2 + st.v_fluid[r,c]**2) for _, st in self.outputs])
            v_avg = np.where(h_tot > 1e-6, (h_s*vs + h_f*vf) / h_tot, 0.0)
            discharge = h_tot * v_avg * self.terrain.cell_size

            self.ax_height.plot(times, h_tot, '-', color=colour, linewidth=1.5, label=label)
            self.ax_vel.plot(times, v_avg, '-', color=colour, linewidth=1.5, label=label)
            self.ax_vel.plot(times, discharge, '--', color=colour, linewidth=1, alpha=0.6)

        self.ax_height.set_ylabel("Flow Height (m)", color='#aaa', fontsize=8)
        self.ax_height.set_title("Hydrograph — Total Flow Height", color='#e8e8e8', fontsize=9)
        self.ax_height.legend(loc='upper right', fontsize=6,
                               facecolor='#2a2a4a', edgecolor='#555', labelcolor='#ddd')
        self._style_ax(self.ax_height)

        self.ax_vel.set_ylabel("Velocity (m/s) / Discharge (m³/s, dashed)", color='#aaa', fontsize=8)
        self.ax_vel.set_xlabel("Time (s)", color='#aaa', fontsize=8)
        self.ax_vel.legend(loc='upper right', fontsize=6,
                            facecolor='#2a2a4a', edgecolor='#555', labelcolor='#ddd')
        self._style_ax(self.ax_vel)

        self.hydro_figure.tight_layout(pad=0.8)
        self.hydro_canvas.draw_idle()

    # ── Events ────────────────────────────────────────────

    def _on_map_click(self, event):
        if self.terrain is None or event.inaxes != self.map_ax:
            return
        col = max(0, min(int(round(event.xdata)), self.terrain.cols - 1))
        row = max(0, min(int(round(event.ydata)), self.terrain.rows - 1))
        self._add_monitor_point(row, col)

    def _on_map_scroll(self, event):
        if event.inaxes != self.map_ax:
            return
        f = 0.85 if event.button == 'up' else 1.15
        xl, yl = self.map_ax.get_xlim(), self.map_ax.get_ylim()
        xm, ym = event.xdata, event.ydata
        self.map_ax.set_xlim([xm + (x - xm)*f for x in xl])
        self.map_ax.set_ylim([ym + (y - ym)*f for y in yl])
        self.map_canvas.draw_idle()

    def _on_set_point(self):
        if self.terrain is None:
            return
        self._monitor_points = []
        self._add_monitor_point(self.row_spin.value(), self.col_spin.value())

    def _on_add_point(self):
        if self.terrain is None:
            return
        self._add_monitor_point(self.row_spin.value(), self.col_spin.value())

    def _add_monitor_point(self, row: int, col: int):
        self._monitor_points.append((row, col))
        self.row_spin.setValue(row); self.col_spin.setValue(col)
        n = len(self._monitor_points)
        self.info_label.setText(f"{n} monitor point{'s' if n>1 else ''} active")
        self._draw_map(); self._draw_hydrograph()

    def _clear_all_points(self):
        self._monitor_points = []
        self.info_label.setText("Click a point on the terrain or enter coordinates below")
        self._draw_map(); self._draw_hydrograph()

    def _save_hydro(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Hydrograph", "hydrograph.png",
            "PNG (*.png);;PDF (*.pdf);;SVG (*.svg)"
        )
        if path:
            self.hydro_figure.savefig(path, dpi=150, bbox_inches='tight',
                                       facecolor=self.hydro_figure.get_facecolor())

    def _style_ax(self, ax):
        ax.tick_params(colors='#888', labelsize=7)
        for sp in ax.spines.values():
            sp.set_color('#3d3d5c')


# ─────────────────────────────────────────────────────────────────────────────
# StatisticsWidget
# ─────────────────────────────────────────────────────────────────────────────

class StatisticsWidget(QWidget):
    """
    Post-simulation statistical analysis panel.

    Sections:
      • Descriptive — flow height distribution
      • Spatial     — area, perimeter, centroid, bounding box
      • Temporal    — peak time, rise time, recession time
      • Phase       — solid/fluid fractions, peak discharge
      • Tests       — Shapiro-Wilk normality, KS vs uniform

    Results displayed as formatted text + exportable as CSV or TXT.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.terrain = None
        self.outputs = None
        self._stats: dict = {}
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        # Button row
        btn_row = QHBoxLayout()
        self.btn_compute = QPushButton("🔄 Compute Statistics")
        self.btn_compute.clicked.connect(self._compute_and_display)
        btn_row.addWidget(self.btn_compute)
        btn_save_csv = QPushButton("💾 Save CSV")
        btn_save_csv.clicked.connect(lambda: self._save_stats('csv'))
        btn_row.addWidget(btn_save_csv)
        btn_save_txt = QPushButton("💾 Save TXT")
        btn_save_txt.clicked.connect(lambda: self._save_stats('txt'))
        btn_row.addWidget(btn_save_txt)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Output text area
        self.text_out = QTextEdit()
        self.text_out.setReadOnly(True)
        self.text_out.setFont(__import__('PyQt6.QtGui', fromlist=['QFont']).QFont("Consolas", 9))
        self.text_out.setStyleSheet(
            "QTextEdit { background: #1a1a2e; color: #d0d0d0; border: 1px solid #3d3d5c; }"
        )
        self.text_out.setPlaceholderText("Run a simulation, then click 'Compute Statistics'.")
        layout.addWidget(self.text_out, stretch=1)

    def set_data(self, terrain, outputs):
        self.terrain = terrain
        self.outputs = outputs
        self._stats = {}
        self.text_out.setPlaceholderText("Data loaded — click 'Compute Statistics'.")
        # Auto compute
        self._compute_and_display()

    def _compute_and_display(self):
        if self.terrain is None or not self.outputs:
            self.text_out.setPlainText("No simulation data available.")
            return
        try:
            self._stats = self._compute_stats()
            self.text_out.setPlainText(self._format_stats(self._stats))
        except Exception as e:
            self.text_out.setPlainText(f"Error computing statistics:\n{e}")

    # ── Core computation ──────────────────────────────────

    def _compute_stats(self) -> dict:
        from scipy import stats as sp_stats

        terrain = self.terrain
        outputs = self.outputs
        cs = terrain.cell_size

        # Build max-height array and time series
        max_h = np.zeros((terrain.rows, terrain.cols))
        times = np.array([t for t, _ in outputs])
        peak_h_time_series = np.array([
            (s.h_solid + s.h_fluid).max() for _, s in outputs
        ])

        for _, s in outputs:
            max_h = np.maximum(max_h, s.h_solid + s.h_fluid)

        active = max_h[max_h > 0.01]
        flow_mask = max_h > 0.01

        # ── Descriptive ──────────────────────────────────
        desc = {}
        if active.size > 0:
            desc['count'] = int(active.size)
            desc['mean']  = float(np.mean(active))
            desc['std']   = float(np.std(active))
            desc['min']   = float(active.min())
            desc['max']   = float(active.max())
            for p in [5, 25, 50, 75, 95]:
                desc[f'p{p}'] = float(np.percentile(active, p))
            from scipy.stats import skew, kurtosis
            desc['skewness'] = float(skew(active))
            desc['kurtosis'] = float(kurtosis(active))
        else:
            desc = {k: 0.0 for k in ['count','mean','std','min','max',
                                       'p5','p25','p50','p75','p95','skewness','kurtosis']}

        # ── Spatial ───────────────────────────────────────
        spatial = {}
        spatial['flow_area_m2'] = float(flow_mask.sum() * cs ** 2)
        spatial['flow_area_km2'] = spatial['flow_area_m2'] / 1e6

        # Perimeter (count boundary cells × cell_size)
        from scipy.ndimage import binary_erosion
        interior = binary_erosion(flow_mask)
        boundary = flow_mask & ~interior
        spatial['perimeter_m'] = float(boundary.sum() * cs)

        # Centroid
        if flow_mask.any():
            coords = np.argwhere(flow_mask)
            cr, cc = float(coords[:, 0].mean()), float(coords[:, 1].mean())
            spatial['centroid_row'] = cr
            spatial['centroid_col'] = cc
            if terrain.is_geographic and terrain.cell_size_deg > 0:
                lat_top = terrain.y_origin + terrain.rows * terrain.cell_size_deg
                spatial['centroid_lat'] = lat_top - cr * terrain.cell_size_deg
                spatial['centroid_lon'] = terrain.x_origin + cc * terrain.cell_size_deg
            rmin, rmax = int(coords[:,0].min()), int(coords[:,0].max())
            cmin, cmax = int(coords[:,1].min()), int(coords[:,1].max())
            spatial['bbox_rows'] = f"{rmin}–{rmax}"
            spatial['bbox_cols'] = f"{cmin}–{cmax}"
            spatial['bbox_h_km'] = (rmax - rmin) * cs / 1000
            spatial['bbox_w_km'] = (cmax - cmin) * cs / 1000
        else:
            spatial['centroid_row'] = 0.0; spatial['centroid_col'] = 0.0
            spatial['centroid_lat'] = 0.0; spatial['centroid_lon'] = 0.0
            spatial['bbox_rows'] = '—'; spatial['bbox_cols'] = '—'
            spatial['bbox_h_km'] = 0.0; spatial['bbox_w_km'] = 0.0

        # ── Temporal ──────────────────────────────────────
        temporal = {}
        temporal['duration_s'] = float(times[-1] - times[0]) if len(times) > 1 else 0.0
        if peak_h_time_series.max() > 0:
            peak_idx = int(np.argmax(peak_h_time_series))
            temporal['peak_height_m'] = float(peak_h_time_series[peak_idx])
            temporal['peak_time_s'] = float(times[peak_idx])
            # Rise time: 0→90% of peak
            thresh90 = 0.9 * temporal['peak_height_m']
            above = np.where(peak_h_time_series >= thresh90)[0]
            temporal['rise_time_s'] = float(times[above[0]]) if above.size else float(times[peak_idx])
            # Recession: time from peak to half-peak
            half_peak = 0.5 * temporal['peak_height_m']
            after_peak = peak_h_time_series[peak_idx:]
            below_half = np.where(after_peak <= half_peak)[0]
            temporal['recession_time_s'] = float(times[peak_idx + below_half[0]]) if below_half.size else float(times[-1])
        else:
            temporal['peak_height_m'] = 0.0; temporal['peak_time_s'] = 0.0
            temporal['rise_time_s'] = 0.0; temporal['recession_time_s'] = 0.0

        # ── Phase ─────────────────────────────────────────
        phase = {}
        solid_fracs = []
        discharges = []
        for _, s in outputs:
            h_tot = s.h_solid + s.h_fluid
            active_mask = h_tot > 0.01
            if active_mask.any():
                sf = s.h_solid[active_mask] / h_tot[active_mask]
                solid_fracs.append(sf.mean())
                v_avg = np.where(h_tot > 1e-6,
                    (s.h_solid * np.sqrt(s.u_solid**2 + s.v_solid**2) +
                     s.h_fluid * np.sqrt(s.u_fluid**2 + s.v_fluid**2)) / h_tot, 0.0)
                discharges.append((h_tot * v_avg * cs)[active_mask].max())
        if solid_fracs:
            phase['mean_solid_fraction'] = float(np.mean(solid_fracs))
            phase['mean_fluid_fraction'] = 1.0 - phase['mean_solid_fraction']
            phase['peak_discharge_m3s'] = float(max(discharges))
        else:
            phase['mean_solid_fraction'] = 0.0
            phase['mean_fluid_fraction'] = 0.0
            phase['peak_discharge_m3s'] = 0.0

        # ── Statistical Tests ─────────────────────────────
        tests = {}
        if active.size >= 3:
            sw_stat, sw_p = sp_stats.shapiro(active[:5000])  # cap for speed
            tests['shapiro_wilk_stat'] = float(sw_stat)
            tests['shapiro_wilk_p'] = float(sw_p)
            tests['shapiro_wilk_normal'] = sw_p > 0.05

            ks_stat, ks_p = sp_stats.kstest(active, 'uniform',
                                              args=(active.min(), active.max() - active.min()))
            tests['ks_uniform_stat'] = float(ks_stat)
            tests['ks_uniform_p'] = float(ks_p)
        else:
            tests = {'shapiro_wilk_stat': 0.0, 'shapiro_wilk_p': 0.0,
                     'shapiro_wilk_normal': False,
                     'ks_uniform_stat': 0.0, 'ks_uniform_p': 0.0}

        return {
            'descriptive': desc, 'spatial': spatial,
            'temporal': temporal, 'phase': phase, 'tests': tests,
            'n_frames': len(outputs), 'n_active_cells': int(flow_mask.sum())
        }

    def _format_stats(self, s: dict) -> str:
        lines = []
        lines.append("═" * 60)
        lines.append("  PyDebFlow v0.2.0 — Statistical Analysis")
        lines.append("═" * 60)

        # Descriptive
        d = s['descriptive']
        lines.append("\n📊 DESCRIPTIVE STATISTICS (Max Flow Height)")
        lines.append(f"  Frames analysed  : {s['n_frames']}")
        lines.append(f"  Active cells     : {s['n_active_cells']}")
        lines.append(f"  Mean             : {d['mean']:.4f} m")
        lines.append(f"  Std Dev          : {d['std']:.4f} m")
        lines.append(f"  Min / Max        : {d['min']:.4f} / {d['max']:.4f} m")
        lines.append(f"  Percentiles (P5/P25/P50/P75/P95):")
        lines.append(f"    {d['p5']:.3f} / {d['p25']:.3f} / {d['p50']:.3f} / {d['p75']:.3f} / {d['p95']:.3f} m")
        lines.append(f"  Skewness         : {d['skewness']:.4f}")
        lines.append(f"  Kurtosis         : {d['kurtosis']:.4f}")

        # Spatial
        sp = s['spatial']
        lines.append("\n🗺️  SPATIAL STATISTICS")
        lines.append(f"  Flow Area        : {sp['flow_area_m2']:,.1f} m²  ({sp['flow_area_km2']:.4f} km²)")
        lines.append(f"  Perimeter        : {sp['perimeter_m']:,.1f} m")
        lines.append(f"  Centroid (row,col): ({sp['centroid_row']:.1f}, {sp['centroid_col']:.1f})")
        if self.terrain and self.terrain.is_geographic:
            lines.append(f"  Centroid (lat,lon): ({sp.get('centroid_lat',0):.5f}°, {sp.get('centroid_lon',0):.5f}°)")
        lines.append(f"  Bounding Box Rows: {sp['bbox_rows']}")
        lines.append(f"  Bounding Box Cols: {sp['bbox_cols']}")
        lines.append(f"  Extent H × W     : {sp['bbox_h_km']:.3f} km × {sp['bbox_w_km']:.3f} km")

        # Temporal
        t = s['temporal']
        lines.append("\n⏱️  TEMPORAL STATISTICS")
        lines.append(f"  Simulation Duration : {t['duration_s']:.1f} s")
        lines.append(f"  Peak Height         : {t['peak_height_m']:.4f} m  at t = {t['peak_time_s']:.1f} s")
        lines.append(f"  Rise Time (→90%)    : {t['rise_time_s']:.1f} s")
        lines.append(f"  Recession Time (→50%): {t['recession_time_s']:.1f} s")

        # Phase
        ph = s['phase']
        lines.append("\n⚗️  PHASE STATISTICS")
        lines.append(f"  Mean Solid Fraction : {ph['mean_solid_fraction']:.4f}  ({ph['mean_solid_fraction']*100:.1f}%)")
        lines.append(f"  Mean Fluid Fraction : {ph['mean_fluid_fraction']:.4f}  ({ph['mean_fluid_fraction']*100:.1f}%)")
        lines.append(f"  Peak Discharge      : {ph['peak_discharge_m3s']:.4f} m³/s")

        # Tests
        ts = s['tests']
        lines.append("\n🔬 STATISTICAL TESTS")
        sw_result = "✅ Likely normal" if ts.get('shapiro_wilk_normal') else "❌ Non-normal"
        lines.append(f"  Shapiro-Wilk (normality):")
        lines.append(f"    W = {ts['shapiro_wilk_stat']:.6f},  p = {ts['shapiro_wilk_p']:.6f}  → {sw_result}")
        lines.append(f"  KS Test (vs Uniform):")
        lines.append(f"    D = {ts['ks_uniform_stat']:.6f},  p = {ts['ks_uniform_p']:.6f}")

        lines.append("\n" + "═" * 60)
        return "\n".join(lines)

    def _save_stats(self, fmt: str):
        if not self._stats:
            QMessageBox.warning(self, "No Data", "Compute statistics first.")
            return
        if fmt == 'csv':
            path, _ = QFileDialog.getSaveFileName(self, "Save Statistics", "statistics.csv", "CSV (*.csv)")
            if path:
                self._save_as_csv(path)
        else:
            path, _ = QFileDialog.getSaveFileName(self, "Save Statistics", "statistics.txt", "Text (*.txt)")
            if path:
                Path(path).write_text(self._format_stats(self._stats), encoding='utf-8')

    def _save_as_csv(self, path: str):
        rows = []
        rows.append("Section,Key,Value")
        for section, vals in self._stats.items():
            if isinstance(vals, dict):
                for k, v in vals.items():
                    rows.append(f"{section},{k},{v}")
            else:
                rows.append(f"summary,{section},{vals}")
        Path(path).write_text("\n".join(rows), encoding='utf-8')
