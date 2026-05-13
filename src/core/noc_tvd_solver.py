"""
NOC-TVD Solver for OpenDebris.

Implements the Non-Oscillatory Central (NOC) scheme with 
Total Variation Diminishing (TVD) flux limiting for numerical stability.
Based on Tai et al. (2002) and Mergili et al. (2017) - r.avaflow.
"""

import math
import warnings
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Callable, Any
from numba import njit, prange
import time

try:
    from numba import cuda
    _CUDA_AVAILABLE = cuda.is_available()
    _CUDA_IMPORT_ERROR = None
except Exception as exc:
    cuda = None
    _CUDA_AVAILABLE = False
    _CUDA_IMPORT_ERROR = exc

if cuda is not None:
    try:
        from numba.cuda.cudadrv.error import (
            CudaAPIError,
            CudaDriverError,
            CudaRuntimeError,
            CudaSupportError,
            NvvmSupportError,
        )

        _CUDA_RUNTIME_ERRORS = (
            CudaAPIError,
            CudaDriverError,
            CudaRuntimeError,
            CudaSupportError,
            NvvmSupportError,
            OSError,
        )
    except Exception:
        _CUDA_RUNTIME_ERRORS = (OSError,)
else:
    _CUDA_RUNTIME_ERRORS = (OSError,)

EPSILON = 1e-10
HEIGHT_EPS = 1e-6
FLUID_VELOCITY_FACTOR = 0.9
CUDA_REFLECT_THREADS = 256


def _ensure_float64(array: np.ndarray) -> np.ndarray:
    return array if array.dtype == np.float64 else array.astype(np.float64)

from .flow_model import FlowState, FlowParameters, TwoPhaseFlowModel
from .terrain import Terrain


@dataclass
class SolverConfig:
    """Configuration for the NOC-TVD solver."""
    
    # Time stepping
    cfl_number: float = 0.4  # CFL condition (< 0.5 for stability)
    max_timestep: float = 0.5  # Maximum allowed timestep (s)
    min_timestep: float = 1e-6  # Minimum timestep (s)
    
    # Flux limiter: 'minmod', 'superbee', 'vanleer', 'none'
    flux_limiter: str = 'minmod'
    
    # Boundary conditions: 'outflow', 'reflective', 'periodic'
    boundary_type: str = 'outflow'
    
    # Stability
    height_threshold: float = 1e-4  # Minimum height for active cells
    velocity_damping: float = 0.0  # Optional artificial viscosity
    
    # Performance
    use_numba: bool = True  # Use Numba JIT acceleration
    use_cuda: bool = False  # Use Numba CUDA acceleration if available
    cuda_block_size: Tuple[int, int] = (16, 16)  # CUDA block size (rows, cols); tune per GPU


@njit(cache=True)
def minmod(a: float, b: float) -> float:
    """Minmod flux limiter."""
    if a * b <= 0:
        return 0.0
    elif abs(a) < abs(b):
        return a
    else:
        return b


@njit(cache=True)
def superbee(a: float, b: float) -> float:
    """Superbee flux limiter."""
    if a * b <= 0:
        return 0.0
    s = 1.0 if a > 0 else -1.0
    return s * max(min(2 * abs(a), abs(b)), min(abs(a), 2 * abs(b)))


@njit(cache=True) 
def vanleer(a: float, b: float) -> float:
    """Van Leer flux limiter."""
    if a * b <= 0:
        return 0.0
    return 2 * a * b / (a + b + EPSILON)


@njit(cache=True, parallel=True)
def compute_fluxes_x(h_solid: np.ndarray, h_fluid: np.ndarray,
                      u_solid: np.ndarray, v_solid: np.ndarray,
                      u_fluid: np.ndarray, v_fluid: np.ndarray,
                      g: float, dx: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute x-direction fluxes with TVD reconstruction.
    """
    rows, cols = h_solid.shape
    
    # Flux arrays (at cell interfaces: cols+1)
    F_hs = np.zeros((rows, cols + 1))
    F_hf = np.zeros((rows, cols + 1))
    F_hu_s = np.zeros((rows, cols + 1))
    F_hu_f = np.zeros((rows, cols + 1))
    
    for i in prange(rows):
        for j in range(cols + 1):
            # Left and right states
            if j == 0:
                hs_L, hs_R = h_solid[i, 0], h_solid[i, 0]
                hf_L, hf_R = h_fluid[i, 0], h_fluid[i, 0]
                us_L, us_R = u_solid[i, 0], u_solid[i, 0]
                uf_L, uf_R = u_fluid[i, 0], u_fluid[i, 0]
            elif j == cols:
                hs_L, hs_R = h_solid[i, -1], h_solid[i, -1]
                hf_L, hf_R = h_fluid[i, -1], h_fluid[i, -1]
                us_L, us_R = u_solid[i, -1], u_solid[i, -1]
                uf_L, uf_R = u_fluid[i, -1], u_fluid[i, -1]
            else:
                hs_L, hs_R = h_solid[i, j-1], h_solid[i, j]
                hf_L, hf_R = h_fluid[i, j-1], h_fluid[i, j]
                us_L, us_R = u_solid[i, j-1], u_solid[i, j]
                uf_L, uf_R = u_fluid[i, j-1], u_fluid[i, j]
            
            # Wave speed estimates
            h_L = hs_L + hf_L
            h_R = hs_R + hf_R
            c_L = np.sqrt(g * max(h_L, 0)) if h_L > 0 else 0
            c_R = np.sqrt(g * max(h_R, 0)) if h_R > 0 else 0
            
            # HLL wave speeds
            s_L = min(us_L - c_L, us_R - c_R, 0)
            s_R = max(us_L + c_L, us_R + c_R, 0)
            
            # Fluxes at left and right
            flux_hs_L = hs_L * us_L
            flux_hs_R = hs_R * us_R
            flux_hf_L = hf_L * uf_L
            flux_hf_R = hf_R * uf_R
            
            flux_hu_s_L = hs_L * us_L**2 + 0.5 * g * hs_L**2
            flux_hu_s_R = hs_R * us_R**2 + 0.5 * g * hs_R**2
            flux_hu_f_L = hf_L * uf_L**2 + 0.5 * g * hf_L**2
            flux_hu_f_R = hf_R * uf_R**2 + 0.5 * g * hf_R**2
            
            # HLL flux
            if s_R - s_L > EPSILON:
                denom = s_R - s_L
                F_hs[i, j] = (s_R * flux_hs_L - s_L * flux_hs_R + s_L * s_R * (hs_R - hs_L)) / denom
                F_hf[i, j] = (s_R * flux_hf_L - s_L * flux_hf_R + s_L * s_R * (hf_R - hf_L)) / denom
                F_hu_s[i, j] = (s_R * flux_hu_s_L - s_L * flux_hu_s_R + s_L * s_R * (hs_R * us_R - hs_L * us_L)) / denom
                F_hu_f[i, j] = (s_R * flux_hu_f_L - s_L * flux_hu_f_R + s_L * s_R * (hf_R * uf_R - hf_L * uf_L)) / denom
            else:
                F_hs[i, j] = 0.5 * (flux_hs_L + flux_hs_R)
                F_hf[i, j] = 0.5 * (flux_hf_L + flux_hf_R)
                F_hu_s[i, j] = 0.5 * (flux_hu_s_L + flux_hu_s_R)
                F_hu_f[i, j] = 0.5 * (flux_hu_f_L + flux_hu_f_R)
    
    return F_hs, F_hf, F_hu_s, F_hu_f


@njit(cache=True, parallel=True)
def compute_fluxes_y(h_solid: np.ndarray, h_fluid: np.ndarray,
                      u_solid: np.ndarray, v_solid: np.ndarray,
                      u_fluid: np.ndarray, v_fluid: np.ndarray,
                      g: float, dy: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute y-direction fluxes with TVD reconstruction.
    """
    rows, cols = h_solid.shape
    
    G_hs = np.zeros((rows + 1, cols))
    G_hf = np.zeros((rows + 1, cols))
    G_hv_s = np.zeros((rows + 1, cols))
    G_hv_f = np.zeros((rows + 1, cols))
    
    for j in prange(cols):
        for i in range(rows + 1):
            if i == 0:
                hs_L, hs_R = h_solid[0, j], h_solid[0, j]
                hf_L, hf_R = h_fluid[0, j], h_fluid[0, j]
                vs_L, vs_R = v_solid[0, j], v_solid[0, j]
                vf_L, vf_R = v_fluid[0, j], v_fluid[0, j]
            elif i == rows:
                hs_L, hs_R = h_solid[-1, j], h_solid[-1, j]
                hf_L, hf_R = h_fluid[-1, j], h_fluid[-1, j]
                vs_L, vs_R = v_solid[-1, j], v_solid[-1, j]
                vf_L, vf_R = v_fluid[-1, j], v_fluid[-1, j]
            else:
                hs_L, hs_R = h_solid[i-1, j], h_solid[i, j]
                hf_L, hf_R = h_fluid[i-1, j], h_fluid[i, j]
                vs_L, vs_R = v_solid[i-1, j], v_solid[i, j]
                vf_L, vf_R = v_fluid[i-1, j], v_fluid[i, j]
            
            h_L = hs_L + hf_L
            h_R = hs_R + hf_R
            c_L = np.sqrt(g * max(h_L, 0)) if h_L > 0 else 0
            c_R = np.sqrt(g * max(h_R, 0)) if h_R > 0 else 0
            
            s_L = min(vs_L - c_L, vs_R - c_R, 0)
            s_R = max(vs_L + c_L, vs_R + c_R, 0)
            
            flux_hs_L = hs_L * vs_L
            flux_hs_R = hs_R * vs_R
            flux_hf_L = hf_L * vf_L
            flux_hf_R = hf_R * vf_R
            
            flux_hv_s_L = hs_L * vs_L**2 + 0.5 * g * hs_L**2
            flux_hv_s_R = hs_R * vs_R**2 + 0.5 * g * hs_R**2
            flux_hv_f_L = hf_L * vf_L**2 + 0.5 * g * hf_L**2
            flux_hv_f_R = hf_R * vf_R**2 + 0.5 * g * hf_R**2
            
            if s_R - s_L > EPSILON:
                denom = s_R - s_L
                G_hs[i, j] = (s_R * flux_hs_L - s_L * flux_hs_R + s_L * s_R * (hs_R - hs_L)) / denom
                G_hf[i, j] = (s_R * flux_hf_L - s_L * flux_hf_R + s_L * s_R * (hf_R - hf_L)) / denom
                G_hv_s[i, j] = (s_R * flux_hv_s_L - s_L * flux_hv_s_R + s_L * s_R * (hs_R * vs_R - hs_L * vs_L)) / denom
                G_hv_f[i, j] = (s_R * flux_hv_f_L - s_L * flux_hv_f_R + s_L * s_R * (hf_R * vf_R - hf_L * vf_L)) / denom
            else:
                G_hs[i, j] = 0.5 * (flux_hs_L + flux_hs_R)
                G_hf[i, j] = 0.5 * (flux_hf_L + flux_hf_R)
                G_hv_s[i, j] = 0.5 * (flux_hv_s_L + flux_hv_s_R)
                G_hv_f[i, j] = 0.5 * (flux_hv_f_L + flux_hv_f_R)
    
    return G_hs, G_hf, G_hv_s, G_hv_f


@dataclass
class _CudaBuffers:
    shape: Tuple[int, int]
    h_solid: Any
    h_fluid: Any
    u_solid: Any
    v_solid: Any
    u_fluid: Any
    v_fluid: Any
    h_solid_x: Any
    h_fluid_x: Any
    u_solid_x: Any
    u_fluid_x: Any
    F_hs: Any
    F_hf: Any
    F_hu_s: Any
    F_hu_f: Any
    G_hs: Any
    G_hf: Any
    G_hv_s: Any
    G_hv_f: Any
    max_speed: Any
    slope_x: Any
    slope_y: Any


if cuda is not None:
    @cuda.jit
    def _cuda_compute_fluxes_x(h_solid, h_fluid,
                               u_solid, v_solid,
                               u_fluid, v_fluid,
                               g, F_hs, F_hf, F_hu_s, F_hu_f):
        i, j = cuda.grid(2)
        rows = h_solid.shape[0]
        cols = h_solid.shape[1]
        if i < rows and j < cols + 1:
            if j == 0:
                hs_L = h_solid[i, 0]
                hs_R = hs_L
                hf_L = h_fluid[i, 0]
                hf_R = hf_L
                us_L = u_solid[i, 0]
                us_R = us_L
                uf_L = u_fluid[i, 0]
                uf_R = uf_L
            elif j == cols:
                hs_L = h_solid[i, cols - 1]
                hs_R = hs_L
                hf_L = h_fluid[i, cols - 1]
                hf_R = hf_L
                us_L = u_solid[i, cols - 1]
                us_R = us_L
                uf_L = u_fluid[i, cols - 1]
                uf_R = uf_L
            else:
                hs_L = h_solid[i, j - 1]
                hs_R = h_solid[i, j]
                hf_L = h_fluid[i, j - 1]
                hf_R = h_fluid[i, j]
                us_L = u_solid[i, j - 1]
                us_R = u_solid[i, j]
                uf_L = u_fluid[i, j - 1]
                uf_R = u_fluid[i, j]

            h_L = hs_L + hf_L
            h_R = hs_R + hf_R
            c_L = math.sqrt(g * h_L) if h_L > EPSILON else 0.0
            c_R = math.sqrt(g * h_R) if h_R > EPSILON else 0.0

            s_L = min(us_L - c_L, us_R - c_R, 0.0)
            s_R = max(us_L + c_L, us_R + c_R, 0.0)

            flux_hs_L = hs_L * us_L
            flux_hs_R = hs_R * us_R
            flux_hf_L = hf_L * uf_L
            flux_hf_R = hf_R * uf_R

            flux_hu_s_L = hs_L * us_L * us_L + 0.5 * g * hs_L * hs_L
            flux_hu_s_R = hs_R * us_R * us_R + 0.5 * g * hs_R * hs_R
            flux_hu_f_L = hf_L * uf_L * uf_L + 0.5 * g * hf_L * hf_L
            flux_hu_f_R = hf_R * uf_R * uf_R + 0.5 * g * hf_R * hf_R

            denom = s_R - s_L
            if denom > EPSILON:
                F_hs[i, j] = (s_R * flux_hs_L - s_L * flux_hs_R + s_L * s_R * (hs_R - hs_L)) / denom
                F_hf[i, j] = (s_R * flux_hf_L - s_L * flux_hf_R + s_L * s_R * (hf_R - hf_L)) / denom
                F_hu_s[i, j] = (s_R * flux_hu_s_L - s_L * flux_hu_s_R + s_L * s_R * (hs_R * us_R - hs_L * us_L)) / denom
                F_hu_f[i, j] = (s_R * flux_hu_f_L - s_L * flux_hu_f_R + s_L * s_R * (hf_R * uf_R - hf_L * uf_L)) / denom
            else:
                F_hs[i, j] = 0.5 * (flux_hs_L + flux_hs_R)
                F_hf[i, j] = 0.5 * (flux_hf_L + flux_hf_R)
                F_hu_s[i, j] = 0.5 * (flux_hu_s_L + flux_hu_s_R)
                F_hu_f[i, j] = 0.5 * (flux_hu_f_L + flux_hu_f_R)


    @cuda.jit
    def _cuda_compute_fluxes_y(h_solid, h_fluid,
                               u_solid, v_solid,
                               u_fluid, v_fluid,
                               g, G_hs, G_hf, G_hv_s, G_hv_f):
        i, j = cuda.grid(2)
        rows = h_solid.shape[0]
        cols = h_solid.shape[1]
        if i < rows + 1 and j < cols:
            if i == 0:
                hs_L = h_solid[0, j]
                hs_R = hs_L
                hf_L = h_fluid[0, j]
                hf_R = hf_L
                vs_L = v_solid[0, j]
                vs_R = vs_L
                vf_L = v_fluid[0, j]
                vf_R = vf_L
            elif i == rows:
                hs_L = h_solid[rows - 1, j]
                hs_R = hs_L
                hf_L = h_fluid[rows - 1, j]
                hf_R = hf_L
                vs_L = v_solid[rows - 1, j]
                vs_R = vs_L
                vf_L = v_fluid[rows - 1, j]
                vf_R = vf_L
            else:
                hs_L = h_solid[i - 1, j]
                hs_R = h_solid[i, j]
                hf_L = h_fluid[i - 1, j]
                hf_R = h_fluid[i, j]
                vs_L = v_solid[i - 1, j]
                vs_R = v_solid[i, j]
                vf_L = v_fluid[i - 1, j]
                vf_R = v_fluid[i, j]

            h_L = hs_L + hf_L
            h_R = hs_R + hf_R
            c_L = math.sqrt(g * h_L) if h_L > EPSILON else 0.0
            c_R = math.sqrt(g * h_R) if h_R > EPSILON else 0.0

            s_L = min(vs_L - c_L, vs_R - c_R, 0.0)
            s_R = max(vs_L + c_L, vs_R + c_R, 0.0)

            flux_hs_L = hs_L * vs_L
            flux_hs_R = hs_R * vs_R
            flux_hf_L = hf_L * vf_L
            flux_hf_R = hf_R * vf_R

            flux_hv_s_L = hs_L * vs_L * vs_L + 0.5 * g * hs_L * hs_L
            flux_hv_s_R = hs_R * vs_R * vs_R + 0.5 * g * hs_R * hs_R
            flux_hv_f_L = hf_L * vf_L * vf_L + 0.5 * g * hf_L * hf_L
            flux_hv_f_R = hf_R * vf_R * vf_R + 0.5 * g * hf_R * hf_R

            denom = s_R - s_L
            if denom > EPSILON:
                G_hs[i, j] = (s_R * flux_hs_L - s_L * flux_hs_R + s_L * s_R * (hs_R - hs_L)) / denom
                G_hf[i, j] = (s_R * flux_hf_L - s_L * flux_hf_R + s_L * s_R * (hf_R - hf_L)) / denom
                G_hv_s[i, j] = (s_R * flux_hv_s_L - s_L * flux_hv_s_R + s_L * s_R * (hs_R * vs_R - hs_L * vs_L)) / denom
                G_hv_f[i, j] = (s_R * flux_hv_f_L - s_L * flux_hv_f_R + s_L * s_R * (hf_R * vf_R - hf_L * vf_L)) / denom
            else:
                G_hs[i, j] = 0.5 * (flux_hs_L + flux_hs_R)
                G_hf[i, j] = 0.5 * (flux_hf_L + flux_hf_R)
                G_hv_s[i, j] = 0.5 * (flux_hv_s_L + flux_hv_s_R)
                G_hv_f[i, j] = 0.5 * (flux_hv_f_L + flux_hv_f_R)


    @cuda.jit
    def _cuda_update_x(h_solid, h_fluid, u_solid, u_fluid,
                       F_hs, F_hf, F_hu_s, F_hu_f, dt_dx,
                       h_solid_x, h_fluid_x, u_solid_x, u_fluid_x):
        i, j = cuda.grid(2)
        rows = h_solid.shape[0]
        cols = h_solid.shape[1]
        if i < rows and j < cols:
            hs = h_solid[i, j]
            hf = h_fluid[i, j]
            h_solid_x[i, j] = hs - dt_dx * (F_hs[i, j + 1] - F_hs[i, j])
            h_fluid_x[i, j] = hf - dt_dx * (F_hf[i, j + 1] - F_hf[i, j])

            hu_s = hs * u_solid[i, j]
            hu_f = hf * u_fluid[i, j]
            hu_s_new = hu_s - dt_dx * (F_hu_s[i, j + 1] - F_hu_s[i, j])
            hu_f_new = hu_f - dt_dx * (F_hu_f[i, j + 1] - F_hu_f[i, j])

            if h_solid_x[i, j] > HEIGHT_EPS:
                u_solid_x[i, j] = hu_s_new / h_solid_x[i, j]
            else:
                u_solid_x[i, j] = 0.0

            if h_fluid_x[i, j] > HEIGHT_EPS:
                u_fluid_x[i, j] = hu_f_new / h_fluid_x[i, j]
            else:
                u_fluid_x[i, j] = 0.0


    @cuda.jit
    def _cuda_update_y(h_solid, h_fluid, v_solid, v_fluid,
                       h_solid_x, h_fluid_x, u_solid_x, u_fluid_x,
                       G_hs, G_hf, G_hv_s, G_hv_f, dt_dy):
        i, j = cuda.grid(2)
        rows = h_solid.shape[0]
        cols = h_solid.shape[1]
        if i < rows and j < cols:
            hs_orig = h_solid[i, j]
            hf_orig = h_fluid[i, j]
            vs_orig = v_solid[i, j]
            vf_orig = v_fluid[i, j]

            hs_new = h_solid_x[i, j] - dt_dy * (G_hs[i + 1, j] - G_hs[i, j])
            hf_new = h_fluid_x[i, j] - dt_dy * (G_hf[i + 1, j] - G_hf[i, j])

            hv_s = hs_orig * vs_orig
            hv_f = hf_orig * vf_orig
            hv_s_new = hv_s - dt_dy * (G_hv_s[i + 1, j] - G_hv_s[i, j])
            hv_f_new = hv_f - dt_dy * (G_hv_f[i + 1, j] - G_hv_f[i, j])

            h_solid[i, j] = hs_new
            h_fluid[i, j] = hf_new
            u_solid[i, j] = u_solid_x[i, j]
            u_fluid[i, j] = u_fluid_x[i, j]

            if hs_new > HEIGHT_EPS:
                v_solid[i, j] = hv_s_new / hs_new
            else:
                v_solid[i, j] = 0.0

            if hf_new > HEIGHT_EPS:
                v_fluid[i, j] = hv_f_new / hf_new
            else:
                v_fluid[i, j] = 0.0


    @cuda.jit
    def _cuda_apply_source_and_clamp(h_solid, h_fluid,
                                     u_solid, v_solid,
                                     u_fluid, v_fluid,
                                     slope_x, slope_y,
                                     g, voellmy_mu, voellmy_xi,
                                     height_threshold, velocity_cap,
                                     min_flow_height, dt):
        i, j = cuda.grid(2)
        rows = h_solid.shape[0]
        cols = h_solid.shape[1]
        if i < rows and j < cols:
            hs = h_solid[i, j]
            hf = h_fluid[i, j]
            h_total = hs + hf

            if h_total > height_threshold:
                speed_s = math.sqrt(u_solid[i, j] * u_solid[i, j] +
                                    v_solid[i, j] * v_solid[i, j]) + EPSILON
                fx_g = g * h_total * slope_x[i, j]
                fy_g = g * h_total * slope_y[i, j]

                tau_c = voellmy_mu * g * h_total
                tau_t = g * speed_s * speed_s / voellmy_xi
                tau_total = tau_c + tau_t

                fx_f = -tau_total * u_solid[i, j] / speed_s
                fy_f = -tau_total * v_solid[i, j] / speed_s

                u = u_solid[i, j] + dt * (fx_g + fx_f)
                v = v_solid[i, j] + dt * (fy_g + fy_f)
            else:
                u = 0.0
                v = 0.0

            if hs < 0.0:
                hs = 0.0
            if hf < 0.0:
                hf = 0.0

            if u > velocity_cap:
                u = velocity_cap
            elif u < -velocity_cap:
                u = -velocity_cap
            if v > velocity_cap:
                v = velocity_cap
            elif v < -velocity_cap:
                v = -velocity_cap

            h_total = hs + hf
            if h_total < min_flow_height:
                u = 0.0
                v = 0.0

            h_solid[i, j] = hs
            h_fluid[i, j] = hf
            u_solid[i, j] = u
            v_solid[i, j] = v
            u_fluid[i, j] = FLUID_VELOCITY_FACTOR * u
            v_fluid[i, j] = FLUID_VELOCITY_FACTOR * v


    @cuda.jit
    def _cuda_compute_max_speed(h_solid, h_fluid,
                                u_solid, v_solid,
                                u_fluid, v_fluid,
                                g, max_speed):
        i, j = cuda.grid(2)
        rows = h_solid.shape[0]
        cols = h_solid.shape[1]
        if i < rows and j < cols:
            h_total = h_solid[i, j] + h_fluid[i, j]
            c = math.sqrt(g * h_total) if h_total > EPSILON else 0.0

            us = math.fabs(u_solid[i, j])
            vs = math.fabs(v_solid[i, j])
            uf = math.fabs(u_fluid[i, j])
            vf = math.fabs(v_fluid[i, j])

            solid_speed = (us if us > vs else vs) + c
            fluid_speed = (uf if uf > vf else vf) + c
            max_speed[i, j] = solid_speed if solid_speed > fluid_speed else fluid_speed


    @cuda.jit
    def _cuda_reflect_u(u_solid):
        i = cuda.grid(1)
        rows = u_solid.shape[0]
        cols = u_solid.shape[1]
        if i < rows:
            u_solid[i, 0] = -u_solid[i, 1]
            u_solid[i, cols - 1] = -u_solid[i, cols - 2]


    @cuda.jit
    def _cuda_reflect_v(v_solid):
        j = cuda.grid(1)
        rows = v_solid.shape[0]
        cols = v_solid.shape[1]
        if j < cols:
            v_solid[0, j] = -v_solid[1, j]
            v_solid[rows - 1, j] = -v_solid[rows - 2, j]


    @cuda.reduce
    def _cuda_max_reduce(a, b):
        return a if a > b else b
else:
    _cuda_max_reduce = None


class NOCTVDSolver:
    """
    NOC-TVD numerical solver for two-phase shallow water equations.
    
    Uses HLL Riemann solver with TVD flux limiters for numerical stability.
    """
    
    def __init__(self, terrain: Terrain, model: TwoPhaseFlowModel,
                 config: Optional[SolverConfig] = None):
        self.terrain = terrain
        self.model = model
        self.config = config or SolverConfig()
        self.g = 9.81
        self._cuda_available = _CUDA_AVAILABLE and cuda is not None
        self._use_cuda = False
        self._cuda_buffers: Optional[_CudaBuffers] = None
        self._cuda_slope_x: Optional[Any] = None
        self._cuda_slope_y: Optional[Any] = None
        self._cuda_slope_shape: Optional[Tuple[int, int]] = None
        
        # Pre-compute for efficiency
        self.dx = terrain.cell_size
        self.dy = terrain.cell_size
        
        if self.config.use_cuda:
            if not self._cuda_available:
                if cuda is None:
                    reason = f"numba.cuda import failed: {_CUDA_IMPORT_ERROR}" if _CUDA_IMPORT_ERROR else "numba.cuda unavailable"
                else:
                    reason = "no CUDA-capable GPU/driver detected"
                warnings.warn(f"CUDA requested but not available ({reason}); falling back to CPU.")
            else:
                self._use_cuda = True
                self._ensure_cuda_context()

    def _disable_cuda(self, reason: str, exc: Optional[BaseException] = None) -> None:
        message = reason
        if exc is not None:
            message = f"{reason} ({exc.__class__.__name__}: {exc})"
        warnings.warn(f"{message}; falling back to CPU.")
        self.release_cuda()
        self._use_cuda = False

    def _ensure_cuda_context(self) -> bool:
        if not self._use_cuda:
            return False
        try:
            cuda.current_context()
        except _CUDA_RUNTIME_ERRORS as exc:
            self._disable_cuda("CUDA initialization failed", exc)
            return False
        return True
    
    def compute_timestep(self, state: FlowState) -> float:
        """
        Compute stable timestep from CFL condition.
        
        dt <= CFL * dx / max(|u| + sqrt(gh))
        """
        h_total = state.h_total
        speed_solid = state.speed_solid
        speed_fluid = state.speed_fluid
        
        # Wave speeds
        c = np.sqrt(self.g * np.maximum(h_total, 0))
        max_speed = np.maximum(
            np.maximum(np.abs(state.u_solid), np.abs(state.v_solid)) + c,
            np.maximum(np.abs(state.u_fluid), np.abs(state.v_fluid)) + c
        )
        
        # Avoid division by zero
        max_wave_speed = max_speed.max()
        if max_wave_speed < EPSILON:
            return self.config.max_timestep
        
        dt = self.config.cfl_number * min(self.dx, self.dy) / max_wave_speed
        
        return self._clip_timestep(dt)

    def _clip_timestep(self, dt: float) -> float:
        if dt < self.config.min_timestep:
            return float(self.config.min_timestep)
        if dt > self.config.max_timestep:
            return float(self.config.max_timestep)
        return float(dt)

    def _cuda_grid(self, rows: int, cols: int) -> Tuple[int, int]:
        block = self.config.cuda_block_size
        return (math.ceil(rows / block[0]), math.ceil(cols / block[1]))

    def _cuda_grid_1d(self, count: int, threads: int) -> int:
        return math.ceil(count / threads)

    def _ensure_cuda_buffers(self, shape: Tuple[int, int]) -> _CudaBuffers:
        if self._cuda_buffers is not None and self._cuda_buffers.shape == shape:
            return self._cuda_buffers

        rows, cols = shape
        if self._cuda_slope_x is None or self._cuda_slope_shape != shape:
            slope_x_host = _ensure_float64(self.terrain.slope_x)
            slope_y_host = _ensure_float64(self.terrain.slope_y)
            self._cuda_slope_x = cuda.to_device(slope_x_host)
            self._cuda_slope_y = cuda.to_device(slope_y_host)
            self._cuda_slope_shape = shape

        slope_x = self._cuda_slope_x
        slope_y = self._cuda_slope_y

        buffers = _CudaBuffers(
            shape=shape,
            h_solid=cuda.device_array(shape, dtype=np.float64),
            h_fluid=cuda.device_array(shape, dtype=np.float64),
            u_solid=cuda.device_array(shape, dtype=np.float64),
            v_solid=cuda.device_array(shape, dtype=np.float64),
            u_fluid=cuda.device_array(shape, dtype=np.float64),
            v_fluid=cuda.device_array(shape, dtype=np.float64),
            h_solid_x=cuda.device_array(shape, dtype=np.float64),
            h_fluid_x=cuda.device_array(shape, dtype=np.float64),
            u_solid_x=cuda.device_array(shape, dtype=np.float64),
            u_fluid_x=cuda.device_array(shape, dtype=np.float64),
            F_hs=cuda.device_array((rows, cols + 1), dtype=np.float64),
            F_hf=cuda.device_array((rows, cols + 1), dtype=np.float64),
            F_hu_s=cuda.device_array((rows, cols + 1), dtype=np.float64),
            F_hu_f=cuda.device_array((rows, cols + 1), dtype=np.float64),
            G_hs=cuda.device_array((rows + 1, cols), dtype=np.float64),
            G_hf=cuda.device_array((rows + 1, cols), dtype=np.float64),
            G_hv_s=cuda.device_array((rows + 1, cols), dtype=np.float64),
            G_hv_f=cuda.device_array((rows + 1, cols), dtype=np.float64),
            max_speed=cuda.device_array(shape, dtype=np.float64),
            slope_x=slope_x,
            slope_y=slope_y,
        )

        self._cuda_buffers = buffers
        return buffers

    def _init_cuda_state(self, state: FlowState) -> _CudaBuffers:
        buffers = self._ensure_cuda_buffers(state.h_solid.shape)
        buffers.h_solid.copy_to_device(_ensure_float64(state.h_solid))
        buffers.h_fluid.copy_to_device(_ensure_float64(state.h_fluid))
        buffers.u_solid.copy_to_device(_ensure_float64(state.u_solid))
        buffers.v_solid.copy_to_device(_ensure_float64(state.v_solid))
        buffers.u_fluid.copy_to_device(_ensure_float64(state.u_fluid))
        buffers.v_fluid.copy_to_device(_ensure_float64(state.v_fluid))
        return buffers

    def _cuda_state_to_host(self, buffers: _CudaBuffers) -> FlowState:
        return FlowState(
            h_solid=buffers.h_solid.copy_to_host(),
            h_fluid=buffers.h_fluid.copy_to_host(),
            u_solid=buffers.u_solid.copy_to_host(),
            v_solid=buffers.v_solid.copy_to_host(),
            u_fluid=buffers.u_fluid.copy_to_host(),
            v_fluid=buffers.v_fluid.copy_to_host(),
        )

    def release_cuda(self) -> None:
        """Release cached CUDA buffers and device arrays."""
        self._cuda_buffers = None
        self._cuda_slope_x = None
        self._cuda_slope_y = None
        self._cuda_slope_shape = None

    def _compute_timestep_cuda(self, buffers: _CudaBuffers) -> float:
        rows, cols = buffers.shape
        block = self.config.cuda_block_size
        grid = self._cuda_grid(rows, cols)

        _cuda_compute_max_speed[grid, block](
            buffers.h_solid, buffers.h_fluid,
            buffers.u_solid, buffers.v_solid,
            buffers.u_fluid, buffers.v_fluid,
            self.g, buffers.max_speed
        )

        max_speed_flat = buffers.max_speed.reshape((buffers.max_speed.size,))
        max_speed = float(_cuda_max_reduce(max_speed_flat))
        if max_speed < EPSILON:
            return self.config.max_timestep

        dt = self.config.cfl_number * min(self.dx, self.dy) / max_speed
        return self._clip_timestep(dt)

    def _cuda_step(self, buffers: _CudaBuffers, dt: float) -> None:
        rows, cols = buffers.shape
        block = self.config.cuda_block_size
        grid_cells = self._cuda_grid(rows, cols)
        grid_flux_x = self._cuda_grid(rows, cols + 1)
        grid_flux_y = self._cuda_grid(rows + 1, cols)

        _cuda_compute_fluxes_x[grid_flux_x, block](
            buffers.h_solid, buffers.h_fluid,
            buffers.u_solid, buffers.v_solid,
            buffers.u_fluid, buffers.v_fluid,
            self.g, buffers.F_hs, buffers.F_hf, buffers.F_hu_s, buffers.F_hu_f
        )

        dt_dx = dt / self.dx
        _cuda_update_x[grid_cells, block](
            buffers.h_solid, buffers.h_fluid,
            buffers.u_solid, buffers.u_fluid,
            buffers.F_hs, buffers.F_hf, buffers.F_hu_s, buffers.F_hu_f, dt_dx,
            buffers.h_solid_x, buffers.h_fluid_x, buffers.u_solid_x, buffers.u_fluid_x
        )

        _cuda_compute_fluxes_y[grid_flux_y, block](
            buffers.h_solid_x, buffers.h_fluid_x,
            buffers.u_solid_x, buffers.v_solid,
            buffers.u_fluid_x, buffers.v_fluid,
            self.g, buffers.G_hs, buffers.G_hf, buffers.G_hv_s, buffers.G_hv_f
        )

        dt_dy = dt / self.dy
        _cuda_update_y[grid_cells, block](
            buffers.h_solid, buffers.h_fluid,
            buffers.v_solid, buffers.v_fluid,
            buffers.h_solid_x, buffers.h_fluid_x,
            buffers.u_solid_x, buffers.u_fluid_x,
            buffers.G_hs, buffers.G_hf, buffers.G_hv_s, buffers.G_hv_f,
            dt_dy
        )

        _cuda_apply_source_and_clamp[grid_cells, block](
            buffers.h_solid, buffers.h_fluid,
            buffers.u_solid, buffers.v_solid,
            buffers.u_fluid, buffers.v_fluid,
            buffers.slope_x, buffers.slope_y,
            self.g, self.model.params.voellmy_mu, self.model.params.voellmy_xi,
            self.config.height_threshold, self.model.params.velocity_cap,
            self.model.params.min_flow_height, dt
        )

        if self.config.boundary_type == 'reflective':
            threads = CUDA_REFLECT_THREADS
            blocks_rows = self._cuda_grid_1d(rows, threads)
            blocks_cols = self._cuda_grid_1d(cols, threads)
            _cuda_reflect_u[blocks_rows, threads](buffers.u_solid)
            _cuda_reflect_v[blocks_cols, threads](buffers.v_solid)
    
    def apply_source_terms(self, state: FlowState, dt: float) -> FlowState:
        """Apply source terms: gravity, friction, drag."""
        new_state = state.copy()
        
        h_total = state.h_total
        active = h_total > self.config.height_threshold
        
        # Gravity
        fx_g = self.g * h_total * self.terrain.slope_x
        fy_g = self.g * h_total * self.terrain.slope_y
        
        # Friction (Voellmy-Salm)
        speed_s = state.speed_solid + EPSILON
        mu = self.model.params.voellmy_mu
        xi = self.model.params.voellmy_xi
        
        # Coulomb term
        tau_c = mu * self.g * h_total
        
        # Turbulent term  
        tau_t = self.g * speed_s**2 / xi
        
        tau_total = tau_c + tau_t
        
        # Friction forces (opposite to velocity)
        fx_f = np.where(active, -tau_total * state.u_solid / speed_s, 0)
        fy_f = np.where(active, -tau_total * state.v_solid / speed_s, 0)
        
        # Update velocities
        new_state.u_solid = np.where(active, state.u_solid + dt * (fx_g + fx_f), 0)
        new_state.v_solid = np.where(active, state.v_solid + dt * (fy_g + fy_f), 0)
        
        # Fluid follows solid (simplified)
        new_state.u_fluid = FLUID_VELOCITY_FACTOR * new_state.u_solid
        new_state.v_fluid = FLUID_VELOCITY_FACTOR * new_state.v_solid
        
        return new_state
    
    def step(self, state: FlowState, dt: float) -> FlowState:
        """
        Perform one time step using dimensional splitting.
        """
        new_state = state.copy()
        
        # X-sweep
        F_hs, F_hf, F_hu_s, F_hu_f = compute_fluxes_x(
            state.h_solid, state.h_fluid,
            state.u_solid, state.v_solid,
            state.u_fluid, state.v_fluid,
            self.g, self.dx
        )
        
        dt_dx = dt / self.dx
        new_state.h_solid = state.h_solid - dt_dx * (F_hs[:, 1:] - F_hs[:, :-1])
        new_state.h_fluid = state.h_fluid - dt_dx * (F_hf[:, 1:] - F_hf[:, :-1])
        
        # Update momentum
        hu_s = state.h_solid * state.u_solid
        hu_f = state.h_fluid * state.u_fluid
        hu_s_new = hu_s - dt_dx * (F_hu_s[:, 1:] - F_hu_s[:, :-1])
        hu_f_new = hu_f - dt_dx * (F_hu_f[:, 1:] - F_hu_f[:, :-1])
        
        # Recover velocity
        with np.errstate(divide='ignore', invalid='ignore'):
            new_state.u_solid = np.where(new_state.h_solid > HEIGHT_EPS, 
                                          hu_s_new / new_state.h_solid, 0)
            new_state.u_fluid = np.where(new_state.h_fluid > HEIGHT_EPS,
                                          hu_f_new / new_state.h_fluid, 0)
        
        # Y-sweep
        G_hs, G_hf, G_hv_s, G_hv_f = compute_fluxes_y(
            new_state.h_solid, new_state.h_fluid,
            new_state.u_solid, new_state.v_solid,
            new_state.u_fluid, new_state.v_fluid,
            self.g, self.dy
        )
        
        dt_dy = dt / self.dy
        new_state.h_solid = new_state.h_solid - dt_dy * (G_hs[1:, :] - G_hs[:-1, :])
        new_state.h_fluid = new_state.h_fluid - dt_dy * (G_hf[1:, :] - G_hf[:-1, :])
        
        hv_s = state.h_solid * state.v_solid
        hv_f = state.h_fluid * state.v_fluid
        hv_s_new = hv_s - dt_dy * (G_hv_s[1:, :] - G_hv_s[:-1, :])
        hv_f_new = hv_f - dt_dy * (G_hv_f[1:, :] - G_hv_f[:-1, :])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            new_state.v_solid = np.where(new_state.h_solid > HEIGHT_EPS,
                                          hv_s_new / new_state.h_solid, 0)
            new_state.v_fluid = np.where(new_state.h_fluid > HEIGHT_EPS,
                                          hv_f_new / new_state.h_fluid, 0)
        
        # Source terms
        new_state = self.apply_source_terms(new_state, dt)
        
        # Clamp values
        new_state.clamp_values(self.model.params)
        
        return new_state
    
    def apply_boundary_conditions(self, state: FlowState) -> FlowState:
        """Apply boundary conditions."""
        if self.config.boundary_type == 'outflow':
            # Zero gradient at boundaries (free outflow)
            pass
        elif self.config.boundary_type == 'reflective':
            state.u_solid[:, 0] = -state.u_solid[:, 1]
            state.u_solid[:, -1] = -state.u_solid[:, -2]
            state.v_solid[0, :] = -state.v_solid[1, :]
            state.v_solid[-1, :] = -state.v_solid[-2, :]
        
        return state
    
    def run_simulation(self, initial_state: FlowState, 
                       t_end: float,
                       output_interval: float = 1.0,
                       progress_callback: Optional[Callable] = None) -> List[Tuple[float, FlowState]]:
        """
        Run the simulation from initial state to t_end.
        
        Args:
            initial_state: Initial flow conditions
            t_end: End time (seconds)
            output_interval: Time interval for saving outputs
            progress_callback: Optional callback(progress, time, step)
            
        Returns:
            List of (time, state) tuples at output intervals
        """
        if self._use_cuda:
            return self._run_simulation_cuda(initial_state, t_end, output_interval, progress_callback)

        outputs = [(0.0, initial_state.copy())]
        state = initial_state.copy()
        
        t = 0.0
        step = 0
        next_output = output_interval
        
        start_time = time.time()
        
        while t < t_end:
            # Compute timestep
            dt = self.compute_timestep(state)
            
            # Don't overshoot end time
            if t + dt > t_end:
                dt = t_end - t
            
            # Perform step
            state = self.step(state, dt)
            state = self.apply_boundary_conditions(state)
            
            t += dt
            step += 1
            
            # Output
            if t >= next_output:
                outputs.append((t, state.copy()))
                next_output += output_interval
            
            # Progress callback
            if progress_callback and step % 10 == 0:
                progress = t / t_end
                progress_callback(progress, t, step)
        
        # Final output
        if outputs[-1][0] < t:
            outputs.append((t, state.copy()))
        
        elapsed = time.time() - start_time
        print(f"\n  Simulation completed in {elapsed:.1f}s ({step} steps)")
        
        return outputs

    def _run_simulation_cuda(self, initial_state: FlowState,
                              t_end: float,
                              output_interval: float,
                              progress_callback: Optional[Callable]) -> List[Tuple[float, FlowState]]:
        outputs = [(0.0, initial_state.copy())]
        buffers = self._init_cuda_state(initial_state)

        t = 0.0
        step = 0
        next_output = output_interval

        start_time = time.time()

        while t < t_end:
            dt = self._compute_timestep_cuda(buffers)

            if t + dt > t_end:
                dt = t_end - t

            self._cuda_step(buffers, dt)

            t += dt
            step += 1

            if t >= next_output:
                outputs.append((t, self._cuda_state_to_host(buffers)))
                next_output += output_interval

            if progress_callback and step % 10 == 0:
                progress = t / t_end
                progress_callback(progress, t, step)

        if outputs[-1][0] < t:
            outputs.append((t, self._cuda_state_to_host(buffers)))

        elapsed = time.time() - start_time
        print(f"\n  Simulation completed in {elapsed:.1f}s ({step} steps)")

        return outputs


# Alias for backward compatibility
Terrain.create_synthetic_slope = Terrain.create_synthetic


def test_solver():
    """Test solver functionality."""
    print("=" * 50)
    print("Testing NOC-TVD Solver")
    print("=" * 50)
    
    # Create terrain
    terrain = Terrain.create_synthetic(rows=30, cols=25, slope_angle=25.0)
    
    # Create model and solver
    params = FlowParameters()
    model = TwoPhaseFlowModel(params)
    config = SolverConfig(cfl_number=0.4)
    solver = NOCTVDSolver(terrain, model, config)
    
    # Initial state
    state = FlowState.zeros((terrain.rows, terrain.cols))
    release = terrain.create_release_zone(5, 12, 3, 2.0)
    state.h_solid = release * 0.7
    state.h_fluid = release * 0.3
    
    initial_volume = (state.h_solid.sum() + state.h_fluid.sum()) * terrain.cell_size**2
    print(f"\n1. Initial volume: {initial_volume:.0f} m³")
    
    # Run short simulation
    print("\n2. Running 3s simulation...")
    outputs = solver.run_simulation(state, t_end=3.0, output_interval=1.0)
    
    print(f"3. Output frames: {len(outputs)}")
    
    _, final = outputs[-1]
    final_volume = (final.h_solid.sum() + final.h_fluid.sum()) * terrain.cell_size**2
    print(f"4. Final volume: {final_volume:.0f} m³")
    print(f"5. Volume change: {(final_volume - initial_volume) / initial_volume * 100:.1f}%")
    
    # Check stability
    assert not np.any(np.isnan(final.h_solid)), "NaN in solid height"
    assert not np.any(np.isnan(final.h_fluid)), "NaN in fluid height"
    assert (final.h_solid >= 0).all(), "Negative solid height"
    
    print("\n✓ Solver tests passed!")
    return True


if __name__ == "__main__":
    test_solver()
