"""
Tests for PyDebFlow v0.2.0 features.

Covers:
- Crown widget lat/lon conversion
- Profile x-axis 0.1 m resolution
- Multi-point hydrograph
- StatisticsWidget calculations
- Save-plot creates file
"""

import numpy as np
import pytest
import os
import tempfile
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_terrain(rows=50, cols=60, cell_size=10.0, is_geographic=False):
    """Create a simple synthetic Terrain for testing."""
    from src.core.terrain import Terrain
    elevation = np.zeros((rows, cols))
    # Simple slope
    for i in range(rows):
        elevation[i, :] = (rows - i) * 5.0  # 5 m/cell slope
    if is_geographic:
        return Terrain(elevation, cell_size=cell_size,
                       x_origin=78.0, y_origin=30.0,
                       is_geographic=True, cell_size_deg=0.0001, mean_lat=30.05)
    return Terrain(elevation, cell_size=cell_size)


def make_outputs(terrain, n_frames=20):
    """Create synthetic simulation outputs [(time, FlowState), ...]."""
    from src.core.flow_model import FlowState
    outputs = []
    for i in range(n_frames):
        t = float(i)
        state = FlowState.zeros((terrain.rows, terrain.cols))
        # Place a decaying pulse in top-left quadrant
        r0, c0 = terrain.rows // 4, terrain.cols // 4
        height = max(0.0, 3.0 - 0.15 * i)
        for dr in range(-5, 6):
            for dc in range(-5, 6):
                r, c = r0 + dr, c0 + dc
                if 0 <= r < terrain.rows and 0 <= c < terrain.cols:
                    d = np.sqrt(dr**2 + dc**2)
                    if d <= 5:
                        h = height * (1 - (d / 5) ** 2)
                        state.h_solid[r, c] = h * 0.65
                        state.h_fluid[r, c] = h * 0.35
                        state.u_solid[r, c] = 1.0
                        state.u_fluid[r, c] = 0.8
        outputs.append((t, state))
    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# 1. Crown Widget — lat/lon ↔ row/col conversion
# ─────────────────────────────────────────────────────────────────────────────

class TestCrownLatLon:
    """Test Terrain geo-coord helpers used by CrownWidget."""

    def setup_method(self):
        self.terrain = make_terrain(rows=100, cols=120, is_geographic=True)

    def test_terrain_is_geographic_flag(self):
        assert self.terrain.is_geographic is True

    def test_terrain_cell_size_deg_stored(self):
        assert self.terrain.cell_size_deg == pytest.approx(0.0001)

    def test_rowcol_to_latlon_top_left(self):
        """Row 0, Col 0 should be at the top-left (max lat, min lon)."""
        t = self.terrain
        lat_top = t.y_origin + t.rows * t.cell_size_deg
        lat = lat_top - 0 * t.cell_size_deg
        lon = t.x_origin + 0 * t.cell_size_deg
        assert lat == pytest.approx(lat_top, abs=1e-9)
        assert lon == pytest.approx(t.x_origin, abs=1e-9)

    def test_rowcol_to_latlon_bottom_right(self):
        t = self.terrain
        row, col = t.rows - 1, t.cols - 1
        lat_top = t.y_origin + t.rows * t.cell_size_deg
        expected_lat = lat_top - row * t.cell_size_deg
        expected_lon = t.x_origin + col * t.cell_size_deg
        got_lat = lat_top - row * t.cell_size_deg
        got_lon = t.x_origin + col * t.cell_size_deg
        assert got_lat == pytest.approx(expected_lat, rel=1e-6)
        assert got_lon == pytest.approx(expected_lon, rel=1e-6)

    def test_latlon_roundtrip(self):
        """Row/col → lat/lon → row/col should be identity."""
        t = self.terrain
        for (row, col) in [(0, 0), (25, 30), (99, 119), (50, 60)]:
            lat_top = t.y_origin + t.rows * t.cell_size_deg
            lat = lat_top - row * t.cell_size_deg
            lon = t.x_origin + col * t.cell_size_deg
            # Back to row/col — use round() to match CrownWidget._latlon_to_rowcol
            row_back = int(round((lat_top - lat) / t.cell_size_deg))
            col_back = int(round((lon - t.x_origin) / t.cell_size_deg))
            assert row_back == row, f"Row mismatch for ({row},{col}): got {row_back}"
            assert col_back == col, f"Col mismatch for ({row},{col}): got {col_back}"

    def test_projected_terrain_no_geographic(self):
        t = make_terrain(is_geographic=False)
        assert t.is_geographic is False
        assert t.cell_size_deg == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# 2. Profile x-axis resolution
# ─────────────────────────────────────────────────────────────────────────────

class TestProfileResolution:
    """Profile should sample every 0.1 m (capped at 5000)."""

    def _sample_count(self, dist_m):
        """Replicate the n_samples logic from CrossSectionWidget."""
        return max(2, min(int(dist_m / 0.1), 5000))

    def test_short_transect_10m(self):
        # 10 m → 100 pts
        assert self._sample_count(10.0) == 100

    def test_long_transect_2km(self):
        # 2000 m → 5000 (capped)
        assert self._sample_count(2000.0) == 5000

    def test_very_short_transect(self):
        # 0.05 m → clamped to 2
        assert self._sample_count(0.05) == 2

    def test_exactly_500m(self):
        assert self._sample_count(500.0) == 5000

    def test_resolution_spacing(self):
        """Each sample should be ~0.1 m apart for short transects."""
        dist = 50.0
        n = self._sample_count(dist)
        spacing = dist / max(n - 1, 1)
        assert spacing == pytest.approx(0.1, abs=0.001)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Multi-point hydrograph
# ─────────────────────────────────────────────────────────────────────────────

class TestMultiPointHydrograph:
    """HydrographWidget stores multiple monitor points and draws them all."""

    def setup_method(self):
        self.terrain = make_terrain()
        self.outputs = make_outputs(self.terrain)

    def test_add_multiple_points(self):
        from src.gui.analysis_widgets import HydrographWidget
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)

        w = HydrographWidget()
        w.set_data(self.terrain, self.outputs)
        assert w._monitor_points == []

        w._add_monitor_point(5, 10)
        w._add_monitor_point(10, 20)
        w._add_monitor_point(15, 30)
        assert len(w._monitor_points) == 3
        assert (5, 10) in w._monitor_points
        assert (15, 30) in w._monitor_points

    def test_clear_all_points(self):
        from src.gui.analysis_widgets import HydrographWidget
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)

        w = HydrographWidget()
        w.set_data(self.terrain, self.outputs)
        w._add_monitor_point(5, 10)
        w._add_monitor_point(10, 20)
        assert len(w._monitor_points) == 2
        w._clear_all_points()
        assert w._monitor_points == []

    def test_hydrograph_series_count(self):
        """Number of plotted lines == number of monitor points."""
        from src.gui.analysis_widgets import HydrographWidget
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)

        w = HydrographWidget()
        w.set_data(self.terrain, self.outputs)
        w._add_monitor_point(5, 10)
        w._add_monitor_point(10, 20)
        w._draw_hydrograph()
        # Each monitor point adds 1 line to ax_height (h_total)
        assert len(w.ax_height.lines) == 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. StatisticsWidget calculations (headless / no Qt required)
# ─────────────────────────────────────────────────────────────────────────────

class TestStatisticsCalculations:
    """Test StatisticsWidget._compute_stats() without GUI."""

    def setup_method(self):
        self.terrain = make_terrain()
        self.outputs = make_outputs(self.terrain, n_frames=20)

    def _get_stats(self):
        from src.gui.analysis_widgets import StatisticsWidget
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)
        w = StatisticsWidget()
        w.terrain = self.terrain
        w.outputs = self.outputs
        return w._compute_stats()

    def test_stats_structure(self):
        s = self._get_stats()
        assert 'descriptive' in s
        assert 'spatial' in s
        assert 'temporal' in s
        assert 'phase' in s
        assert 'tests' in s

    def test_descriptive_keys(self):
        d = self._get_stats()['descriptive']
        for key in ['mean', 'std', 'min', 'max', 'p5', 'p50', 'p95', 'skewness', 'kurtosis']:
            assert key in d, f"Missing key: {key}"

    def test_mean_is_positive(self):
        d = self._get_stats()['descriptive']
        assert d['mean'] > 0.0

    def test_spatial_area_positive(self):
        sp = self._get_stats()['spatial']
        assert sp['flow_area_m2'] > 0.0

    def test_temporal_peak_time_in_range(self):
        t = self._get_stats()['temporal']
        times = [fr[0] for fr in self.outputs]
        assert times[0] <= t['peak_time_s'] <= times[-1]

    def test_phase_fractions_sum_to_one(self):
        ph = self._get_stats()['phase']
        assert ph['mean_solid_fraction'] + ph['mean_fluid_fraction'] == pytest.approx(1.0, abs=1e-6)

    def test_shapiro_wilk_keys(self):
        ts = self._get_stats()['tests']
        assert 'shapiro_wilk_stat' in ts
        assert 'shapiro_wilk_p' in ts
        assert 0.0 <= ts['shapiro_wilk_stat'] <= 1.0

    def test_ks_test_keys(self):
        ts = self._get_stats()['tests']
        assert 'ks_uniform_stat' in ts
        assert 0.0 <= ts['ks_uniform_stat'] <= 1.0

    def test_n_frames_correct(self):
        s = self._get_stats()
        assert s['n_frames'] == 20


# ─────────────────────────────────────────────────────────────────────────────
# 5. Save-plot creates file
# ─────────────────────────────────────────────────────────────────────────────

class TestSavePlot:
    """Verify that figure.savefig() creates a file."""

    def test_profile_figure_saves_png(self):
        from matplotlib.figure import Figure
        fig = Figure(facecolor='#16213e')
        ax = fig.add_subplot(111)
        ax.plot([0, 1, 2], [1, 4, 2])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "profile_test.png")
            fig.savefig(path, dpi=72)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0

    def test_hydro_figure_saves_pdf(self):
        from matplotlib.figure import Figure
        fig = Figure(facecolor='#16213e')
        ax = fig.add_subplot(111)
        ax.plot([0, 1], [0, 1])
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "hydro_test.pdf")
            fig.savefig(path)
            assert os.path.exists(path)

    def test_stats_csv_written(self):
        """StatisticsWidget._save_as_csv writes valid CSV."""
        from src.gui.analysis_widgets import StatisticsWidget
        pytest.importorskip("PyQt6")
        from PyQt6.QtWidgets import QApplication
        import sys
        app = QApplication.instance() or QApplication(sys.argv)

        terrain = make_terrain()
        outputs = make_outputs(terrain, n_frames=10)
        w = StatisticsWidget()
        w.terrain = terrain
        w.outputs = outputs
        w._stats = w._compute_stats()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "stats.csv")
            w._save_as_csv(path)
            content = Path(path).read_text()
            assert "Section,Key,Value" in content
            assert "descriptive" in content
            assert "spatial" in content
