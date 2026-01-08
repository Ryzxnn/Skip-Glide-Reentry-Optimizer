"""Skip-reentry optimizer for Orion-class CEV from lunar-return conditions.

This script simulates a 3-DOF point-mass skip reentry and optimizes a bank
angle (roll) schedule to maximize downrange while respecting peak g-load and
convective heating limits. Run directly to perform the optimization and
produce plots/CSV outputs. Only SciPy, NumPy, Matplotlib, and the standard
library are required.

The model intentionally remains lightweight and transparent so it can be read
and modified by students. See the Config dataclass for the main parameters.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Literal

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution, minimize

# Toggle to skip the global optimization for quick testing
RUN_OPTIMIZATION = True
RUN_TRADE_STUDIES = False


@dataclass
class ValidationSpec:
    target_range_km: float = 8890.0
    target_skip_apogee_km: float = 99.0
    target_peak_g: float = 4.0
    w_range: float = 1.0
    w_apogee: float = 0.5
    w_g: float = 0.5
    range_scale_km: float = 500.0
    apogee_scale_km: float = 30.0
    g_scale: float = 0.5
    apogee_required: bool = True


@dataclass(frozen=True)
class VehicleSpec:
    key: str
    name: str
    mass_kg: float
    diameter_m: float
    nose_radius_m: float
    aero_model: Literal["const", "table"] = "table"
    L_over_D: float = 0.30
    C_D_const: float = 1.0
    aero_mach_table: Tuple[float, ...] = (2, 5, 10, 15, 25)
    aero_cd_table: Tuple[float, ...] = (0.9, 1.0, 1.05, 1.05, 1.05)
    aero_ld_table: Tuple[float, ...] = (0.05, 0.20, 0.32, 0.32, 0.30)
    C_sg: Optional[float] = None
    notes: str = ""

    def area_ref_m2(self) -> float:
        return math.pi * (self.diameter_m * 0.5) ** 2


# Default capsule presets (approximate; refine Apollo values from primary sources as needed)
VEHICLE_PRESETS: Dict[str, VehicleSpec] = {
    "orion": VehicleSpec(
        key="orion",
        name="Orion CM (approx)",
        mass_kg=9200.0,
        diameter_m=5.0,
        nose_radius_m=1.0,
        aero_model="table",
        aero_mach_table=(2, 5, 10, 15, 25),
        aero_cd_table=(0.9, 1.0, 1.05, 1.05, 1.05),
        aero_ld_table=(0.05, 0.20, 0.32, 0.32, 0.30),
        notes="Baseline Orion-like capsule.",
    ),
    "apollo": VehicleSpec(
        key="apollo",
        name="Apollo CM (approx)",
        mass_kg=5560.0,  # CM-only mass; keep separate from CSM
        diameter_m=3.9,
        nose_radius_m=1.0,
        aero_model="table",
        aero_mach_table=(2, 5, 10, 15, 25),
        aero_cd_table=(0.9, 1.0, 1.05, 1.05, 1.05),
        aero_ld_table=(0.04, 0.17, 0.28, 0.30, 0.28),  # slightly lower L/D curve vs Orion baseline
        notes="Apollo CM-only approx; refine later with primary sources.",
    ),
}

VALIDATION_PRESETS: Dict[str, ValidationSpec] = {
    "orion": ValidationSpec(
        target_range_km=8890.0,
        target_skip_apogee_km=99.0,
        target_peak_g=4.0,
        w_range=1.0,
        w_apogee=0.5,
        w_g=0.5,
        range_scale_km=500.0,
        apogee_scale_km=30.0,
        g_scale=0.5,
        apogee_required=True,
    ),
    "apollo": ValidationSpec(
        target_range_km=8890.0,
        target_skip_apogee_km=99.0,
        target_peak_g=4.0,
        w_range=1.0,
        w_apogee=0.5,
        w_g=0.5,
        range_scale_km=500.0,
        apogee_scale_km=30.0,
        g_scale=0.5,
        apogee_required=True,
    ),
}


@dataclass
class Config:
    """Simulation and optimization configuration."""

    # Vehicle and environment
    vehicle: VehicleSpec = field(default_factory=lambda: VEHICLE_PRESETS["orion"])
    m: float = 9200.0  # kg (set by apply_vehicle)
    diameter_m: float = 5.0  # m (set by apply_vehicle)
    R_e: float = 6_371_000.0  # m
    mu: float = 3.986_004_418e14  # m^3/s^2
    A_ref: float = math.pi * (2.5**2)  # m^2, updated via vehicle
    L_over_D: float = 0.30  # set via vehicle when aero_model const
    C_D_const: float = 1.0  # set via vehicle when aero_model const
    use_table_aero: bool = False  # deprecated; set via vehicle
    mach_table: Optional[List[Tuple[float, float, float]]] = None  # (Mach, C_D, C_L) set via vehicle
    aero_model: str = "table"  # "table" or "const"

    # Atmosphere
    rho0: float = 1.225  # kg/m^3
    H: float = 7200.0  # scale height, m

    # Heating (Sutton-Graves)
    R_n: float = 1.0  # nose radius, m
    C_sg: float = 1.83e-4

    # Initial conditions
    h0: float = 120_000.0  # m
    V0: float = 11_000.0  # m/s
    gamma0_deg: float = -5.5  # deg (negative means descending)
    theta0: float = 0.0
    Q0: float = 0.0

    # Optimization / control
    K: int = 7  # number of bank angle knots
    t_end_guess: float = 1800.0  # s
    t_max: float = 3000.0  # s
    sigma_bounds: Tuple[float, float] = (0.0, 180.0)  # deg
    de_maxiter: int = 12
    de_popsize: int = 8
    de_log_every: int = 2
    de_tol: float = 0.0
    de_polish: bool = False
    de_workers: int = -1
    de_updating: str = "deferred"
    de_walltime_s: float = 180.0

    # Constraints
    g_limit: float = 4.5  # g's
    qdot_limit_MW: float = 10.0  # MW/m^2

    # Termination
    V_terminal: float = 500.0  # m/s
    h_terminal: float = 20_000.0  # m
    h_escape: float = 200_000.0  # m

    # Penalty weights (scale vs meters of downrange objective)
    w_g: float = 1e9
    w_q: float = 1e7
    infeasible_penalty: float = 1e12

    # Validation bounds (for regression checks)
    target_final_range_bounds_km: Tuple[float, float] = (8500.0, 9500.0)
    target_skip_apogee_bounds_km: Tuple[float, float] = (85.0, 130.0)
    target_peak_g_bounds: Tuple[float, float] = (3.5, 4.5)

    # Additional penalties
    require_skip: bool = True
    w_skip: float = 1e11
    enable_range_cap: bool = True
    max_range_km: float = 9500.0
    w_range: float = 1e3
    event_margin: float = 1.02
    enable_must_dip: bool = True
    qdyn_min_kPa: float = 5.0
    w_dip: float = 1e9

    # Atmosphere crossing thresholds for skip detection
    rho_threshold_in: float = 2e-6
    rho_threshold_out: float = 5e-7

    # Aero Mach-dependent tables
    aero_mach_table: Tuple[float, ...] = (2, 5, 10, 15, 25)
    aero_cd_table: Tuple[float, ...] = (0.9, 1.0, 1.05, 1.05, 1.05)
    aero_ld_table: Tuple[float, ...] = (0.05, 0.20, 0.32, 0.32, 0.30)

    # Study mode settings
    study_use_slsqp_only: bool = True
    study_slsqp_maxiter: int = 60
    study_de_maxiter: int = 6
    study_de_popsize: int = 8

    # ODE fidelity controls
    ode_rtol_de: float = 1e-4
    ode_atol_de: float = 1e-7
    ode_max_step_de: float = 10.0
    ode_rtol_final: float = 1e-6
    ode_atol_final: float = 1e-9
    ode_max_step_final: float = 2.0

    # Validation objective tuning
    validation_mode: bool = False
    validation: ValidationSpec = field(default_factory=ValidationSpec)

    # Solver settings
    rtol: float = 1e-6
    atol: float = 1e-9
    max_step: float = 2.0

    # Optional fixed bank profile for quick testing when RUN_OPTIMIZATION=False
    sigma_default: Tuple[float, ...] = (0, 0, 0, 180, 180, 120, 90)

    def t_knots(self) -> np.ndarray:
        return np.linspace(0.0, self.t_end_guess, self.K)

    def compute_A_ref(self) -> float:
        return math.pi * (self.diameter_m / 2.0) ** 2

    def apply_vehicle(self, v: VehicleSpec) -> None:
        """Apply vehicle parameters to config (single source of truth)."""
        self.vehicle = v
        self.m = v.mass_kg
        self.diameter_m = v.diameter_m
        self.A_ref = v.area_ref_m2()
        self.R_n = v.nose_radius_m

        # Heating override
        if v.C_sg is not None:
            self.C_sg = v.C_sg

        # Aero selection
        self.aero_model = v.aero_model
        if v.aero_model == "table":
            self.use_table_aero = True
            self.mach_table = None
            self.aero_mach_table = v.aero_mach_table
            self.aero_cd_table = v.aero_cd_table
            self.aero_ld_table = v.aero_ld_table
        else:
            self.use_table_aero = False
            self.L_over_D = v.L_over_D
            self.C_D_const = v.C_D_const


def ensure_results_dir(base: str = "results") -> str:
    path = os.path.join(base)
    os.makedirs(path, exist_ok=True)
    return path


def load_vehicle_from_json(path: str) -> VehicleSpec:
    with open(path, "r") as f:
        data = json.load(f)
    if "key" not in data:
        data["key"] = os.path.splitext(os.path.basename(path))[0].lower()
    return VehicleSpec(**data)


def load_warmstart(path: str) -> Optional[np.ndarray]:
    """Load saved sigma knots if available."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
        sigma = data.get("sigma_knots_deg")
        if sigma is None:
            return None
        return np.array(sigma, dtype=float)
    except Exception:
        return None


def save_warmstart(
    path: str,
    t_knots: np.ndarray,
    sigma_knots: np.ndarray,
    res: Dict,
    mode: str,
    cfg: Config,
    vehicle_key: str = "",
    vehicle_name: str = "",
) -> None:
    payload = {
        "mode": mode,
        "vehicle_key": vehicle_key,
        "vehicle_name": vehicle_name,
        "t_knots_s": [float(x) for x in t_knots],
        "sigma_knots_deg": [float(x) for x in sigma_knots],
        "metrics": {
            "range_km": res["final_range"] / 1000.0,
            "skip_apogee_km": (res["skip_apogee"] / 1000.0) if res["skip_apogee"] is not None else None,
            "peak_g": res["peak_g"],
            "peak_qdot_MWm2": res["peak_qdot"] / 1e6,
            "total_heat_MJm2": res["total_heat"] / 1e6,
            "time_s": float(res["t"][-1]) if res.get("t") is not None and len(res["t"]) else None,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "config_snapshot": {
            "g_limit": cfg.g_limit,
            "qdot_limit_MW": cfg.qdot_limit_MW,
            "require_skip": cfg.require_skip,
            "max_range_km": cfg.max_range_km,
        },
        "vehicle_params": {
            "m": cfg.m,
            "diameter_m": cfg.diameter_m,
            "A_ref": cfg.A_ref,
            "R_n": cfg.R_n,
        },
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

def atmosphere_density(h: np.ndarray, cfg: Config) -> np.ndarray:
    """Simple exponential atmosphere with clamping at very low density."""
    rho = cfg.rho0 * np.exp(-np.maximum(h, 0.0) / cfg.H)
    return np.clip(rho, 1e-12, None)


def speed_of_sound(h: np.ndarray) -> np.ndarray:
    """Piecewise-linear speed of sound approximation."""
    h = np.asarray(h)
    a = np.empty_like(h, dtype=float)
    a[h <= 11_000.0] = 340.0
    a[h >= 20_000.0] = 295.0
    mask = (h > 11_000.0) & (h < 20_000.0)
    a[mask] = np.interp(h[mask], [11_000.0, 20_000.0], [340.0, 295.0])
    return a


def mach_number(V: np.ndarray, h: np.ndarray) -> np.ndarray:
    return V / speed_of_sound(h)


def print_atmosphere_sanity(cfg: Config) -> None:
    """Print a small table of density values for diagnostics."""
    alts = [60_000, 80_000, 100_000, 120_000]
    print("Atmosphere sanity (rho):")
    for alt in alts:
        rho = atmosphere_density(np.array([alt], dtype=float), cfg)[0]
        print(f"  {alt/1000:5.0f} km: {rho:.3e} kg/m^3")
    print(f"Aero model: {cfg.aero_model}, using mach_table: {cfg.mach_table is not None}")


def interpolate_table(x: float, table: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """Piecewise-linear interpolation for (Mach -> C_D, C_L)."""
    mach_vals = [row[0] for row in table]
    cd_vals = [row[1] for row in table]
    cl_vals = [row[2] for row in table]
    cd = np.interp(x, mach_vals, cd_vals)
    cl = np.interp(x, mach_vals, cl_vals)
    return float(cd), float(cl)


def aero_coeffs(Mach: float, cfg: Config) -> Tuple[float, float]:
    """Aerodynamic coefficients with deterministic precedence."""
    if cfg.aero_model == "const":
        cd = cfg.C_D_const
        cl = cfg.C_D_const * cfg.L_over_D
        return cd, cl

    # Table mode
    if cfg.mach_table is not None:
        cd, cl = interpolate_table(Mach, cfg.mach_table)
    else:
        machs = np.asarray(cfg.aero_mach_table, dtype=float)
        cds = np.asarray(cfg.aero_cd_table, dtype=float)
        lds = np.asarray(cfg.aero_ld_table, dtype=float)
        m_clamped = np.clip(Mach, machs.min(), machs.max())
        cd = float(np.interp(m_clamped, machs, cds))
        ld = float(np.interp(m_clamped, machs, lds))
        cl = cd * ld

    cd = max(cd, 0.05)
    cl = max(cl, 0.0)
    return cd, cl


def qdot_convective(h: np.ndarray, V: np.ndarray, cfg: Config) -> np.ndarray:
    """Sutton–Graves convective stagnation-point correlation."""
    rho = atmosphere_density(h, cfg)
    return cfg.C_sg * np.sqrt(rho / cfg.R_n) * V**3


def bank_angle_deg(
    t: np.ndarray, sigma_knots_deg: Iterable[float], cfg: Config, t_knots: Optional[np.ndarray] = None
) -> np.ndarray:
    """Piecewise-linear bank schedule sigma(t) in degrees."""
    if t_knots is None:
        t_knots = cfg.t_knots()
    return np.interp(t, t_knots, sigma_knots_deg)


def equations_of_motion(
    t: float, y: np.ndarray, cfg: Config, sigma_knots_deg: Iterable[float], t_knots: np.ndarray
) -> np.ndarray:
    """3-DOF point-mass equations of motion."""
    h, theta, V, gamma, Q_total = y
    r = cfg.R_e + h
    g = cfg.mu / (r**2)
    rho = atmosphere_density(h, cfg)
    qbar = 0.5 * rho * V**2
    Mach = V / speed_of_sound(h)
    cd, cl = aero_coeffs(Mach, cfg)
    D = qbar * cd * cfg.A_ref
    L = qbar * cl * cfg.A_ref

    sigma_rad = math.radians(float(np.interp(t, t_knots, sigma_knots_deg)))
    L_v = L * math.cos(sigma_rad)

    # Guard against divide-by-zero if V gets very small near the end
    V_safe = max(V, 1.0)

    dhdt = V * math.sin(gamma)
    dthetadt = (V * math.cos(gamma)) / r
    dVdt = -(D / cfg.m) - g * math.sin(gamma)
    dgamdt = (L_v / (cfg.m * V_safe)) + (V * math.cos(gamma)) / r - (g * math.cos(gamma)) / V_safe
    dQdt = qdot_convective(h, V, cfg)
    return np.array([dhdt, dthetadt, dVdt, dgamdt, dQdt], dtype=float)


def terminal_event(cfg: Config) -> Callable:
    def event(t, y):
        h, _, V, _, _ = y
        return max(h - cfg.h_terminal, V - cfg.V_terminal)

    event.terminal = True
    event.direction = -1
    return event


def ground_event(t, y):
    h = y[0]
    return h


ground_event.terminal = True
ground_event.direction = -1


def g_limit_event(cfg: Config) -> Callable:
    def event(t, y):
        h, _, V, _, _ = y
        rho = atmosphere_density(h, cfg)
        qbar = 0.5 * rho * V**2
        Mach = V / speed_of_sound(h)
        cd, cl = aero_coeffs(Mach, cfg)
        D = qbar * cd * cfg.A_ref
        L = qbar * cl * cfg.A_ref
        g_load = math.sqrt(L**2 + D**2) / (cfg.m * 9.80665)
        return cfg.g_limit * cfg.event_margin - g_load

    event.terminal = True
    event.direction = -1
    return event


def qdot_limit_event(cfg: Config) -> Callable:
    def event(t, y):
        h, _, V, _, _ = y
        qdot = qdot_convective(h, V, cfg)
        return cfg.qdot_limit_MW * 1e6 * cfg.event_margin - qdot

    event.terminal = True
    event.direction = -1
    return event


def simulate_profile(
    sigma_knots_deg: Iterable[float], cfg: Config, cache: Optional[Dict] = None, fidelity: str = "final"
) -> Dict:
    """Simulate one bank-angle profile. Returns a results dictionary."""
    sigma_knots_deg = tuple(float(s) for s in sigma_knots_deg)
    cache_key = (fidelity,) + tuple(round(s, 3) for s in sigma_knots_deg)
    if cache is not None and cache_key in cache:
        return cache[cache_key]

    t_knots = cfg.t_knots()
    y0 = np.array(
        [
            cfg.h0,
            cfg.theta0,
            cfg.V0,
            math.radians(cfg.gamma0_deg),
            cfg.Q0,
        ],
        dtype=float,
    )

    if fidelity == "de":
        rtol = cfg.ode_rtol_de
        atol = cfg.ode_atol_de
        max_step = cfg.ode_max_step_de
    else:
        rtol = cfg.ode_rtol_final
        atol = cfg.ode_atol_final
        max_step = cfg.ode_max_step_final

    events = [g_limit_event(cfg), qdot_limit_event(cfg), terminal_event(cfg), ground_event]
    sol = solve_ivp(
        fun=lambda t, y: equations_of_motion(t, y, cfg, sigma_knots_deg, t_knots),
        t_span=(0.0, cfg.t_max),
        y0=y0,
        events=events,
        max_step=max_step,
        rtol=rtol,
        atol=atol,
    )

    event_names = ["g_limit", "qdot_limit", "terminal", "ground"]
    event_hit = "max_time"
    if sol.t_events:
        for name, arr in zip(event_names, sol.t_events):
            if len(arr) > 0:
                event_hit = name
                break

    result = {"feasible": True, "reason": "", "event": event_hit}

    t = sol.t
    h, theta, V, gamma, Q_total = sol.y

    if t.size == 0:
        result.update(
            {
                "feasible": False,
                "reason": result.get("reason", "integration failure"),
                "t": t,
                "h": h,
                "theta": theta,
                "V": V,
                "gamma": gamma,
                "Q_total": Q_total,
            }
        )
        if cache is not None:
            cache[cache_key] = result
        return result

    if sol.status == -1 or sol.y.shape[1] == 0:
        result.update({"feasible": False, "reason": "integration failure"})
    # Escape condition
    if np.max(h) > cfg.h_escape and sol.y[2, -1] > 7800.0:
        result.update({"feasible": False, "reason": "escape/infeasible"})
    if event_hit in ("g_limit", "qdot_limit"):
        result.update({"feasible": False, "reason": f"constraint_violation:{event_hit}"})

    # Derived quantities
    sigma_series = bank_angle_deg(t, sigma_knots_deg, cfg, t_knots)
    rho = atmosphere_density(h, cfg)
    qbar = 0.5 * rho * V**2
    Mach = mach_number(V, h)
    cds, cls = np.vectorize(lambda M: aero_coeffs(M, cfg))(Mach)
    D = qbar * cds * cfg.A_ref
    L = qbar * cls * cfg.A_ref
    g_load = np.sqrt(L**2 + D**2) / (cfg.m * 9.80665)
    qdot = qdot_convective(h, V, cfg)

    peak_qdot = float(np.max(qdot))
    peak_qdot_MW = peak_qdot / 1e6
    peak_g = float(np.max(g_load))
    peak_qdyn_Pa = float(np.max(qbar))
    peak_qdyn_kPa = peak_qdyn_Pa / 1000.0
    final_range = float(cfg.R_e * theta[-1])
    total_heat = float(Q_total[-1])

    def find_first_local_min(arr: np.ndarray, start_idx: int) -> Optional[int]:
        for i in range(max(start_idx + 1, 1), len(arr) - 1):
            if arr[i - 1] > arr[i] <= arr[i + 1]:
                return i
        return None

    def find_first_local_max(arr: np.ndarray, start_idx: int) -> Optional[int]:
        for i in range(max(start_idx + 1, 1), len(arr) - 1):
            if arr[i - 1] < arr[i] >= arr[i + 1]:
                return i
        return None

    skip_apogee = None
    skip_apogee_idx = None
    first_dip_idx = None
    second_entry_idx = None

    # Density-based skip detection with hysteresis
    rho_in = cfg.rho_threshold_in
    rho_out = cfg.rho_threshold_out
    in1 = np.where(rho >= rho_in)[0]
    if in1.size > 0:
        first_in_idx = int(in1[0])
        out_idx_candidates = np.where((np.arange(len(rho)) > first_in_idx) & (rho <= rho_out))[0]
        if out_idx_candidates.size > 0:
            out_idx = int(out_idx_candidates[0])
            in2_candidates = np.where((np.arange(len(rho)) > out_idx) & (rho >= rho_in))[0]
            if in2_candidates.size > 0:
                second_entry_idx = int(in2_candidates[0])
                # first dip between first_in and out
                seg_min_idx = find_first_local_min(h, first_in_idx) or int(np.argmin(h[first_in_idx:out_idx + 1])) + first_in_idx
                first_dip_idx = seg_min_idx
                apogee_rel_idx = int(np.argmax(h[out_idx:second_entry_idx + 1]))
                apogee_idx = out_idx + apogee_rel_idx
                min_val = float(np.min(h[first_in_idx:second_entry_idx + 1]))
                apogee_val = float(h[apogee_idx])
                if apogee_val > min_val + 5000.0 and apogee_val > 80_000.0:
                    skip_apogee = apogee_val
                    skip_apogee_idx = apogee_idx

    # Fallback turning-point method if needed
    if skip_apogee is None and h.size > 2:
        below = np.where(h < 90_000.0)[0]
        if below.size > 0:
            start = below[0]
            min_idx = find_first_local_min(h, start) or int(np.argmin(h[start:])) + start
            min_val = h[min_idx]
            max_idx = find_first_local_max(h, min_idx) or int(np.argmax(h[min_idx:])) + min_idx
            max_val = h[max_idx]
            if max_val > min_val + 5000.0 and max_val > 80_000.0:
                skip_apogee = float(max_val)
                skip_apogee_idx = int(max_idx)
            if first_dip_idx is None:
                first_dip_idx = int(min_idx)
    if first_dip_idx is None:
        first_dip_idx = 0 if h.size > 0 else None

    # Event markers for peaks (top two)
    peak_q_indices = np.argsort(-qdot)
    peak_q_times = [t[int(idx)] for idx in peak_q_indices[:2]] if peak_q_indices.size else [None, None]
    peak_g_indices = np.argsort(-g_load)
    peak_g_times = [t[int(idx)] for idx in peak_g_indices[:2]] if peak_g_indices.size else [None, None]

    first_dip_time = float(t[first_dip_idx]) if (first_dip_idx is not None and t.size) else None
    first_dip_alt_km = float(h[first_dip_idx] / 1000.0) if (first_dip_idx is not None and h.size) else None
    skip_apogee_time = float(t[skip_apogee_idx]) if skip_apogee_idx is not None else None
    skip_apogee_km = float(skip_apogee / 1000.0) if skip_apogee is not None else None
    second_entry_time = float(t[second_entry_idx]) if second_entry_idx is not None else None

    result.update(
        {
            "t": t,
            "h": h,
            "theta": theta,
            "V": V,
            "gamma": gamma,
            "Q_total": Q_total,
            "sigma": sigma_series,
            "rho": rho,
            "qbar": qbar,
            "qdot": qdot,
            "g_load": g_load,
            "peak_qdot": peak_qdot,
            "peak_qdot_Wm2": peak_qdot,
            "peak_qdot_MWm2": peak_qdot_MW,
            "peak_g": peak_g,
            "peak_qdyn_Pa": peak_qdyn_Pa,
            "peak_qdyn_kPa": peak_qdyn_kPa,
            "final_range": final_range,
            "total_heat": total_heat,
            "skip_apogee": skip_apogee,
            "skip_apogee_time_s": skip_apogee_time,
            "skip_apogee_alt_km": skip_apogee_km,
            "skip_apogee_idx": skip_apogee_idx,
            "first_dip_idx": first_dip_idx,
            "first_dip_time_s": first_dip_time,
            "first_dip_alt_km": first_dip_alt_km,
            "second_entry_time_s": second_entry_time,
            "peak1_qdot_time_s": float(peak_q_times[0]) if peak_q_times[0] is not None else None,
            "peak2_qdot_time_s": float(peak_q_times[1]) if len(peak_q_times) > 1 and peak_q_times[1] is not None else None,
            "peak1_g_time_s": float(peak_g_times[0]) if peak_g_times[0] is not None else None,
            "peak2_g_time_s": float(peak_g_times[1]) if len(peak_g_times) > 1 and peak_g_times[1] is not None else None,
            "event": result["event"],
        }
    )

    if cache is not None:
        cache[cache_key] = result
    return result


def objective(
    sigma_knots_deg: Iterable[float],
    cfg: Config,
    cache: Dict,
    fidelity: str = "final",
    objective_mode: str = "maxrange",
) -> float:
    """Objective for optimizer: choose between max-range and validation modes."""
    res = simulate_profile(sigma_knots_deg, cfg, cache, fidelity=fidelity)
    range_val = res.get("final_range", float("nan"))
    try:
        range_km = float(range_val) / 1000.0
    except Exception:
        range_km = float("nan")
    if not res["feasible"] or not math.isfinite(range_km):
        return cfg.infeasible_penalty

    penalty = 0.0
    if res["peak_g"] > cfg.g_limit:
        penalty += cfg.w_g * (res["peak_g"] - cfg.g_limit) ** 2
    if res["peak_qdot"] > cfg.qdot_limit_MW * 1e6:
        penalty += cfg.w_q * (res["peak_qdot"] - cfg.qdot_limit_MW * 1e6) ** 2
    if cfg.require_skip and res["skip_apogee"] is None:
        penalty += cfg.w_skip
    if cfg.enable_range_cap and range_km > cfg.max_range_km:
        penalty += cfg.w_range * (range_km - cfg.max_range_km) ** 2
    if cfg.enable_must_dip:
        peak_qdyn_kPa = res.get("peak_qdyn_kPa", 0.0)
        if peak_qdyn_kPa < cfg.qdyn_min_kPa:
            dv = cfg.qdyn_min_kPa - peak_qdyn_kPa
            penalty += cfg.w_dip * dv**2

    # Penalize if integration ended without terminal event (still high alt/vel)
    if res["event"] == "max_time" and (res["h"][-1] > cfg.h_terminal or res["V"][-1] > cfg.V_terminal):
        penalty += cfg.infeasible_penalty * 0.1

    if objective_mode == "validation":
        skip_apogee_km = res["skip_apogee"] / 1000.0 if res["skip_apogee"] is not None else None
        if skip_apogee_km is None and cfg.validation.apogee_required:
            return cfg.infeasible_penalty
        dr = (range_km - cfg.validation.target_range_km) / cfg.validation.range_scale_km
        da = (
            (skip_apogee_km - cfg.validation.target_skip_apogee_km) / cfg.validation.apogee_scale_km
            if skip_apogee_km is not None
            else 10.0
        )
        dg = (res["peak_g"] - cfg.validation.target_peak_g) / cfg.validation.g_scale
        J = cfg.validation.w_range * dr**2 + cfg.validation.w_apogee * da**2 + cfg.validation.w_g * dg**2
        return J + penalty

    return -res["final_range"] + penalty


def validate_artemis_like(res: Dict, cfg: Config) -> Dict:
    """Run simple regression checks against Artemis-like markers."""
    checks: Dict[str, Dict] = {}
    final_range_km = res["final_range"] / 1000.0
    skip_apogee_km = res["skip_apogee"] / 1000.0 if res["skip_apogee"] is not None else None
    peak_g = res["peak_g"]
    peak_qdot_MW = res.get("peak_qdot_MWm2", res["peak_qdot"] / 1e6)
    # Ranges
    fr_lo, fr_hi = cfg.target_final_range_bounds_km
    sa_lo, sa_hi = cfg.target_skip_apogee_bounds_km
    g_lo, g_hi = cfg.target_peak_g_bounds
    q_hi = min(cfg.qdot_limit_MW, 10.0)

    checks["final_range_km"] = {
        "value": final_range_km,
        "range": (fr_lo, fr_hi),
        "pass": fr_lo <= final_range_km <= fr_hi,
    }
    checks["skip_apogee_km"] = {
        "value": skip_apogee_km,
        "range": (sa_lo, sa_hi),
        "pass": skip_apogee_km is not None and sa_lo <= skip_apogee_km <= sa_hi,
    }
    checks["peak_g"] = {
        "value": peak_g,
        "range": (g_lo, g_hi),
        "pass": g_lo <= peak_g <= g_hi,
    }
    checks["peak_qdot_MW"] = {
        "value": peak_qdot_MW,
        "range": (0.5, q_hi),
        "pass": 0.5 <= peak_qdot_MW <= q_hi,
    }
    checks["event_terminal"] = {
        "value": res.get("event", ""),
        "expected": "terminal",
        "pass": res.get("event", "") == "terminal",
    }
    overall = all(item["pass"] for item in checks.values())
    return {"passed": overall, "checks": checks}


def run_optimization(
    cfg: Config, warmstart_sigma: Optional[np.ndarray], objective_mode: str
) -> Tuple[np.ndarray, Dict]:
    """Run differential evolution then SLSQP refinement."""
    bounds = [cfg.sigma_bounds] * cfg.K
    cache: Dict = {}

    def differential_evo(start_sigma: Optional[np.ndarray] = None) -> np.ndarray:
        print("Starting differential evolution...")
        import time

        start_time = time.perf_counter()
        iter_count = [0]
        best_val = [float("inf")]
        best_x = [None]
        infeas = cfg.infeasible_penalty * 0.9

        def callback(xk, convergence):
            iter_count[0] += 1
            val = objective(xk, cfg, cache, fidelity="de", objective_mode=objective_mode)  # coarse fidelity
            if val < best_val[0]:
                best_val[0] = val
                best_x[0] = np.copy(xk)
            if iter_count[0] % cfg.de_log_every == 0 or iter_count[0] == 1:
                elapsed = time.perf_counter() - start_time
                if best_val[0] >= infeas:
                    msg = f"  DE gen {iter_count[0]:02d}, best infeasible ({best_val[0]:.2e}), elapsed {elapsed:.1f}s"
                else:
                    est_range_km = -best_val[0] / 1000.0 if not cfg.validation_mode else float("nan")
                    msg = f"  DE gen {iter_count[0]:02d}, best objective {best_val[0]:.3e}"
                    if not cfg.validation_mode:
                        msg += f" (~{est_range_km:.1f} km)"
                    msg += f", elapsed {elapsed:.1f}s"
                print(msg)
            if time.perf_counter() - start_time > cfg.de_walltime_s:
                print("  Stopping DE early due to wall-time budget.")
                return True
            return False

        workers = cfg.de_workers
        updating = cfg.de_updating
        if os.name == "nt":
            workers = 1
            updating = "immediate"
        elif os.name != "nt" and __name__ != "__main__":
            workers = 1
            updating = "immediate"
        elif workers == 1:
            updating = "immediate"

        init = "latinhypercube"
        if start_sigma is not None:
            base = np.array(start_sigma, dtype=float)
            noises = np.random.normal(0, 5.0, size=(max(4, cfg.de_popsize - 1), cfg.K))
            pop = [np.clip(base + n, cfg.sigma_bounds[0], cfg.sigma_bounds[1]) for n in noises]
            init = np.vstack([base, pop])

        de_result = differential_evolution(
            func=lambda x: objective(x, cfg, cache, fidelity="de", objective_mode=objective_mode),
            bounds=bounds,
            maxiter=cfg.de_maxiter,
            popsize=cfg.de_popsize,
            polish=cfg.de_polish,
            tol=cfg.de_tol,
            disp=False,
            seed=3,
            callback=callback,
            updating=updating,
            workers=workers,
            init=init,
        )
        best_sigma = de_result.x if de_result.success else (best_x[0] if best_x[0] is not None else de_result.x)
        return best_sigma

    best_sigma_de = differential_evo(warmstart_sigma)

    # Optional local refine with SLSQP
    print("Refining with SLSQP...")
    x0 = warmstart_sigma if warmstart_sigma is not None else best_sigma_de
    local = minimize(
        fun=lambda x: objective(x, cfg, cache, fidelity="final", objective_mode=objective_mode),
        x0=x0,
        bounds=bounds,
        method="SLSQP",
        options={"maxiter": 60, "ftol": 1e-3, "disp": False},
    )
    # Evaluate candidates with final fidelity
    best_sigma = best_sigma_de
    best_result = simulate_profile(best_sigma, cfg, cache, fidelity="final")
    best_obj = objective(best_sigma, cfg, cache, fidelity="final", objective_mode=objective_mode)

    if local.success:
        res_local = simulate_profile(local.x, cfg, cache, fidelity="final")
        obj_local = objective(local.x, cfg, cache, fidelity="final", objective_mode=objective_mode)
        if obj_local < best_obj:
            best_sigma, best_result, best_obj = local.x, res_local, obj_local

    return best_sigma, best_result


def run_trade_studies(cfg: Config, base_best_sigma: np.ndarray, base_res: Dict) -> None:
    """Explore constraint trade space with quick sweeps and Pareto plots."""
    print("\n=== Trade Studies (sweeps) ===")
    results_dir = ensure_results_dir()

    def optimize_for_cfg(cfg_mod: Config, x0: np.ndarray) -> Tuple[np.ndarray, Dict]:
        bounds = [cfg_mod.sigma_bounds] * cfg_mod.K
        cache: Dict = {}

        def run_slsqp(start):
            local = minimize(
                fun=lambda x: objective(x, cfg_mod, cache, fidelity="final"),
                x0=start,
                bounds=bounds,
                method="SLSQP",
                options={"maxiter": cfg_mod.study_slsqp_maxiter, "ftol": 1e-3, "disp": False},
            )
            sim = simulate_profile(local.x, cfg_mod, cache)
            return local, sim

        best_sigma = np.array(x0, dtype=float)
        best_res = simulate_profile(best_sigma, cfg_mod, cache)
        best_obj = objective(best_sigma, cfg_mod, cache)

        local, sim_local = run_slsqp(best_sigma)
        if local.success:
            obj_local = objective(local.x, cfg_mod, cache)
            if obj_local < best_obj:
                best_obj, best_sigma, best_res = obj_local, local.x, sim_local

        valid = best_res["feasible"] and (best_res["skip_apogee"] is not None or not cfg_mod.require_skip)
        if not valid:
            # DE rescue with small budget
            de_res = differential_evolution(
                func=lambda x: objective(x, cfg_mod, cache, fidelity="de"),
                bounds=bounds,
                maxiter=cfg_mod.study_de_maxiter,
                popsize=cfg_mod.study_de_popsize,
                polish=False,
                tol=1e-3,
                disp=False,
                seed=7,
                workers=1,
                updating="deferred",
            )
            de_sigma = de_res.x
            sim_de = simulate_profile(de_sigma, cfg_mod, cache)
            obj_de = objective(de_sigma, cfg_mod, cache)
            if obj_de < best_obj:
                best_obj, best_sigma, best_res = obj_de, de_sigma, sim_de
            # Optional polish
            pol, sim_pol = run_slsqp(best_sigma)
            if pol.success:
                obj_pol = objective(pol.x, cfg_mod, cache)
                if obj_pol < best_obj:
                    best_obj, best_sigma, best_res = obj_pol, pol.x, sim_pol
        return best_sigma, best_res

    rows: List[Dict] = []
    qdot_limits = [3, 5, 7.5, 10, 15, 20]
    g_limits = [3.5, 4.0, 4.5, 5.0]

    for q_lim in qdot_limits:
        cfg_mod = copy.deepcopy(cfg)
        cfg_mod.qdot_limit_MW = q_lim
        cfg_mod.de_maxiter = cfg.study_de_maxiter
        cfg_mod.de_popsize = cfg.study_de_popsize
        sigma, res = optimize_for_cfg(cfg_mod, base_best_sigma)
        rows.append(
            {
                "study_type": "qdot_sweep",
                "constraint_value": q_lim,
                "range_km": res["final_range"] / 1000.0,
                "skip_apogee_km": (res["skip_apogee"] / 1000.0) if res["skip_apogee"] is not None else None,
                "peak_g": res["peak_g"],
                "peak_qdot_MWm2": res["peak_qdot"] / 1e6,
                "total_heat_MJm2": res["total_heat"] / 1e6,
                "feasible": bool(res["feasible"] and res["skip_apogee"] is not None),
                "event": res.get("event", ""),
            }
        )

    for g_lim in g_limits:
        cfg_mod = copy.deepcopy(cfg)
        cfg_mod.g_limit = g_lim
        cfg_mod.de_maxiter = cfg.study_de_maxiter
        cfg_mod.de_popsize = cfg.study_de_popsize
        sigma, res = optimize_for_cfg(cfg_mod, base_best_sigma)
        rows.append(
            {
                "study_type": "g_sweep",
                "constraint_value": g_lim,
                "range_km": res["final_range"] / 1000.0,
                "skip_apogee_km": (res["skip_apogee"] / 1000.0) if res["skip_apogee"] is not None else None,
                "peak_g": res["peak_g"],
                "peak_qdot_MWm2": res["peak_qdot"] / 1e6,
                "total_heat_MJm2": res["total_heat"] / 1e6,
                "feasible": bool(res["feasible"] and res["skip_apogee"] is not None),
                "event": res.get("event", ""),
            }
        )

    csv_path = os.path.join(results_dir, "trade_study.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "study_type",
                "constraint_value",
                "range_km",
                "skip_apogee_km",
                "peak_g",
                "peak_qdot_MWm2",
                "total_heat_MJm2",
                "feasible",
                "event",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Trade study results saved to {csv_path}")

    def plot_sweep(study_type: str, filename: str, xlabel: str):
        subset = [r for r in rows if r["study_type"] == study_type]
        if not subset:
            return
        x = [r["constraint_value"] for r in subset]
        y = [r["range_km"] for r in subset]
        fig, ax = plt.subplots()
        ax.plot(x, y, marker="o")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Range (km)")
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(results_dir, filename), dpi=200)

    plot_sweep("qdot_sweep", "study_range_vs_qdotlimit.png", "qdot_limit (MW/m^2)")
    plot_sweep("g_sweep", "study_range_vs_glimit.png", "g_limit (g)")

    # Pareto plots
    figp1, axp1 = plt.subplots()
    axp1.scatter([r["peak_qdot_MWm2"] for r in rows], [r["range_km"] for r in rows], c="tab:blue")
    axp1.set_xlabel("Peak qdot (MW/m^2)")
    axp1.set_ylabel("Range (km)")
    axp1.grid(True)
    figp1.tight_layout()
    figp1.savefig(os.path.join(results_dir, "pareto_range_vs_peakqdot.png"), dpi=200)

    figp2, axp2 = plt.subplots()
    axp2.scatter([r["peak_g"] for r in rows], [r["range_km"] for r in rows], c="tab:orange")
    axp2.set_xlabel("Peak g")
    axp2.set_ylabel("Range (km)")
    axp2.grid(True)
    figp2.tight_layout()
    figp2.savefig(os.path.join(results_dir, "pareto_range_vs_peakg.png"), dpi=200)

    # Summaries
    def top_three(study_type: str):
        subset = [r for r in rows if r["study_type"] == study_type]
        subset.sort(key=lambda x: x["range_km"], reverse=True)
        return subset[:3]

    print("Top ranges (qdot sweep):")
    for r in top_three("qdot_sweep"):
        print(
            f"  qdot_limit={r['constraint_value']} -> range {r['range_km']:.1f} km, peak_g {r['peak_g']:.2f}, peak_qdot {r['peak_qdot_MWm2']:.2f}, skip_apogee {r['skip_apogee_km']}"
        )
    print("Top ranges (g sweep):")
    for r in top_three("g_sweep"):
        print(
            f"  g_limit={r['constraint_value']} -> range {r['range_km']:.1f} km, peak_g {r['peak_g']:.2f}, peak_qdot {r['peak_qdot_MWm2']:.2f}, skip_apogee {r['skip_apogee_km']}"
        )


def run_mode(
    mode_name: str,
    cfg: Config,
    objective_mode: str,
    out_dir: str,
    warmstart_path: Optional[str] = None,
    seed_sigma_knots_deg: Optional[np.ndarray] = None,
    vehicle_key: str = "orion",
    vehicle_name: str = "",
) -> Tuple[np.ndarray, Dict]:
    """Run one optimization mode (validation or maxrange) and write outputs to out_dir."""
    print(f"\n===== RUN MODE: {mode_name} =====")
    print(
        f"Guardrails: must_dip={cfg.enable_must_dip} (qdyn_min={cfg.qdyn_min_kPa} kPa), "
        f"range_cap={cfg.enable_range_cap} (max_range={cfg.max_range_km} km)"
    )
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "vehicle.json"), "w") as f:
        json.dump(asdict(cfg.vehicle), f, indent=2)
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump({"vehicle_key": cfg.vehicle.key, "mode": mode_name}, f, indent=2)

    warm_sigma = seed_sigma_knots_deg
    if warm_sigma is None and warmstart_path:
        warm_sigma = load_warmstart(warmstart_path)
        if warm_sigma is not None:
            print(f"Loaded warmstart from {warmstart_path}")

    best_sigma, res = run_optimization(cfg, warm_sigma, objective_mode)

    print_report(res, best_sigma, cfg)
    reg = validate_artemis_like(res, cfg)
    print("\n=== ARTEMIS-LIKE REGRESSION ===")
    print(f"Overall: {'PASS' if reg['passed'] else 'FAIL'}")
    for name, chk in reg["checks"].items():
        expected = chk.get("range") or chk.get("expected")
        status = "PASS" if chk["pass"] else "FAIL"
        print(f"- {name}: {chk['value']} expected {expected} -> {status}")

    reg_path = os.path.join(out_dir, "artemis_regression.json")
    with open(reg_path, "w") as f:
        json.dump(reg, f, indent=2)

    csv_path = os.path.join(out_dir, "trajectory.csv")
    save_csv(res, csv_path, cfg)
    plot_results(res, cfg, out_dir)
    best_json_path = os.path.join(out_dir, "best_sigma_knots_deg.json")
    save_warmstart(
        best_json_path,
        cfg.t_knots(),
        np.array(best_sigma, dtype=float),
        res,
        objective_mode,
        cfg,
        vehicle_key=vehicle_key,
        vehicle_name=vehicle_name,
    )
    print(f"Saved trajectory, plots, and best profile to {out_dir}")

    return np.array(best_sigma, dtype=float), res


def save_csv(res: Dict, path: str, cfg: Config) -> None:
    """Save trajectory history to CSV."""
    header = [
        "t_s",
        "h_m",
        "downrange_km",
        "V_m_per_s",
        "gamma_deg",
        "sigma_deg",
        "rho_kg_per_m3",
        "qdot_MW_per_m2",
        "g_load",
        "Q_total_MJ_per_m2",
    ]
    meta_lines = [
        f"# vehicle_key={cfg.vehicle.key}",
        f"# vehicle_name={cfg.vehicle.name}",
        f"# mass_kg={cfg.m}",
        f"# diameter_m={cfg.diameter_m}",
        f"# A_ref_m2={cfg.A_ref}",
        f"# R_n_m={cfg.R_n}",
    ]
    rows = zip(
        res["t"],
        res["h"],
        res["theta"] * cfg.R_e / 1000.0,
        res["V"],
        np.degrees(res["gamma"]),
        res["sigma"],
        res["rho"],
        res["qdot"] / 1e6,
        res["g_load"],
        res["Q_total"] / 1e6,
    )
    with open(path, "w", newline="") as f:
        for line in meta_lines:
            f.write(line + "\n")
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def plot_results(res: Dict, cfg: Config, out_dir: str, prefix: str = "skip_reentry") -> None:
    """Generate and save requested plots into out_dir."""
    t = res["t"]
    h_km = res["h"] / 1000.0
    downrange_km = res["theta"] * cfg.R_e / 1000.0
    gamma_deg = np.degrees(res["gamma"])
    sigma_deg = res["sigma"]

    markers = [
        ("Dip", res.get("first_dip_time_s"), "tab:blue"),
        ("Apogee", res.get("skip_apogee_time_s"), "tab:orange"),
        ("2nd Entry", res.get("second_entry_time_s"), "tab:green"),
        ("qdot1", res.get("peak1_qdot_time_s"), "tab:red"),
        ("qdot2", res.get("peak2_qdot_time_s"), "tab:pink"),
        ("g1", res.get("peak1_g_time_s"), "tab:purple"),
        ("g2", res.get("peak2_g_time_s"), "tab:brown"),
    ]

    def add_vlines(ax):
        for label, tm, color in markers:
            if tm is None:
                continue
            ax.axvline(tm, color=color, linestyle="--", alpha=0.6)
            ylim = ax.get_ylim()
            ax.text(tm, ylim[1], label, rotation=90, va="top", ha="right", fontsize=8, color=color)

    def add_point(ax, tm, label, color):
        if tm is None or t.size == 0:
            return
        dr = float(np.interp(tm, t, downrange_km))
        alt = float(np.interp(tm, t, h_km))
        ax.scatter(dr, alt, color=color, s=30, label=label)
        ax.text(dr, alt, label, fontsize=8, color=color)

    def save_show(fig, name):
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, name), dpi=200)

    fig1, ax1 = plt.subplots()
    ax1.plot(downrange_km, h_km)
    fig1.suptitle(cfg.vehicle.name)
    ax1.set_xlabel("Downrange (km)")
    ax1.set_ylabel("Altitude (km)")
    ax1.grid(True)
    for label, tm, color in markers:
        add_point(ax1, tm, label, color)
    save_show(fig1, "altitude_vs_downrange.png")

    fig2, ax2 = plt.subplots()
    ax2.plot(t, h_km)
    fig2.suptitle(cfg.vehicle.name)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Altitude (km)")
    ax2.grid(True)
    add_vlines(ax2)
    save_show(fig2, "altitude_vs_time.png")

    fig3, ax3 = plt.subplots()
    ax3.plot(t, res["V"] / 1000.0)
    fig3.suptitle(cfg.vehicle.name)
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Velocity (km/s)")
    ax3.grid(True)
    add_vlines(ax3)
    save_show(fig3, "velocity_vs_time.png")

    fig4, ax4 = plt.subplots()
    ax4.plot(t, gamma_deg)
    fig4.suptitle(cfg.vehicle.name)
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Flight-path angle (deg)")
    ax4.grid(True)
    add_vlines(ax4)
    save_show(fig4, "gamma_vs_time.png")

    fig5, ax5 = plt.subplots()
    ax5.plot(t, sigma_deg)
    fig5.suptitle(cfg.vehicle.name)
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Bank angle sigma (deg)")
    ax5.grid(True)
    add_vlines(ax5)
    save_show(fig5, "bank_vs_time.png")

    fig6, ax6 = plt.subplots()
    ax6.plot(t, res["qdot"] / 1e6)
    fig6.suptitle(cfg.vehicle.name)
    ax6.set_xlabel("Time (s)")
    ax6.set_ylabel("Convective heat flux (MW/m^2)")
    ax6.grid(True)
    add_vlines(ax6)
    save_show(fig6, "heatflux_vs_time.png")

    fig7, ax7 = plt.subplots()
    ax7.plot(t, res["g_load"])
    fig7.suptitle(cfg.vehicle.name)
    ax7.set_xlabel("Time (s)")
    ax7.set_ylabel("Load factor (g)")
    ax7.grid(True)
    add_vlines(ax7)
    save_show(fig7, "gload_vs_time.png")

    fig8, ax8 = plt.subplots()
    ax8.plot(t, res["qbar"] / 1000.0)
    fig8.suptitle(cfg.vehicle.name)
    ax8.set_xlabel("Time (s)")
    ax8.set_ylabel("Dynamic pressure (kPa)")
    ax8.grid(True)
    add_vlines(ax8)
    save_show(fig8, "qdyn_vs_time.png")

    fig9, ax9 = plt.subplots()
    ax9.plot(t, res["Q_total"] / 1e6)
    fig9.suptitle(cfg.vehicle.name)
    ax9.set_xlabel("Time (s)")
    ax9.set_ylabel("Heat load Q (MJ/m^2)")
    ax9.grid(True)
    add_vlines(ax9)
    save_show(fig9, "heatload_vs_time.png")

    plt.show()


def print_report(res: Dict, sigma_knots: Iterable[float], cfg: Config) -> None:
    """Print key metrics and validation deltas."""
    downrange_km = res["final_range"] / 1000.0
    peak_g = res["peak_g"]
    peak_qdot_MW = res["peak_qdot"] / 1e6
    total_heat_MJ = res["total_heat"] / 1e6
    skip_apogee_km = res["skip_apogee"] / 1000.0 if res["skip_apogee"] is not None else None
    time_final = res["t"][-1]

    print("\n=== Optimal Bank Profile ===")
    print("Times (s):", np.round(cfg.t_knots(), 1))
    print("Sigma knots (deg):", np.round(sigma_knots, 2))

    print("\n=== Trajectory Metrics ===")
    print(f"Final downrange: {downrange_km:,.1f} km")
    if skip_apogee_km is not None:
        print(f"Skip apogee: {skip_apogee_km:,.1f} km")
    else:
        print("Skip apogee: not detected")
    print(f"Peak g-load: {peak_g:.2f} g")
    print(f"Peak heat flux: {peak_qdot_MW:.2f} MW/m^2")
    print(f"Total heat load: {total_heat_MJ:.2f} MJ/m^2")
    print(f"Time to termination: {time_final:.1f} s")
    print(f"Event: {res.get('event', '')} (reason: {res.get('reason', '')})")

    # Validation deltas
    print("\n=== Validation deltas vs Artemis I markers ===")
    dr_delta = downrange_km - cfg.validation.target_range_km
    apogee_delta = (
        skip_apogee_km - cfg.validation.target_skip_apogee_km if skip_apogee_km is not None else float("nan")
    )
    g_delta = peak_g - cfg.validation.target_peak_g
    print(f"Downrange delta: {dr_delta:+.1f} km")
    if skip_apogee_km is not None:
        print(f"Skip apogee delta: {apogee_delta:+.1f} km")
    else:
        print("Skip apogee delta: N/A (no skip detected)")
    print(f"Peak g delta: {g_delta:+.2f} g")


def main():
    parser = argparse.ArgumentParser(description="Skip reentry optimizer (validation and max-range modes)")
    parser.add_argument("--trade-studies", action="store_true", help="Run constraint sweeps")
    parser.add_argument("--no-warmstart", action="store_true", help="Disable warmstart from saved profile")
    parser.add_argument("--run-validation", action="store_true", help="Run validation mode only (unless run-both)")
    parser.add_argument("--run-max-range", action="store_true", help="Run max-range mode only (unless run-both)")
    parser.add_argument("--run-both", action="store_true", help="Run both modes")
    parser.add_argument("--results-dir", type=str, default="results", help="Base results directory")
    parser.add_argument("--seed-bank-json", type=str, help="Path to sigma knots JSON to seed optimization")
    parser.add_argument(
        "--vehicle",
        type=str,
        default="orion",
        choices=sorted(list(VEHICLE_PRESETS.keys()) + ["custom"]),
        help="Vehicle preset key (orion, apollo, custom)",
    )
    parser.add_argument("--vehicle-json", type=str, default=None, help="Path to custom vehicle JSON when --vehicle=custom")
    args = parser.parse_args()

    cfg = Config()

    global RUN_TRADE_STUDIES
    RUN_TRADE_STUDIES = RUN_TRADE_STUDIES or args.trade_studies

    vehicle_key = args.vehicle.lower()
    if args.vehicle_json:
        vehicle_spec = load_vehicle_from_json(args.vehicle_json)
    else:
        if vehicle_key not in VEHICLE_PRESETS:
            raise ValueError(f"Unknown vehicle '{vehicle_key}'. Options: {list(VEHICLE_PRESETS.keys())} or use --vehicle-json")
        vehicle_spec = VEHICLE_PRESETS[vehicle_key]
    vehicle_key = vehicle_spec.key
    cfg.apply_vehicle(vehicle_spec)

    print_atmosphere_sanity(cfg)
    print(
        f"Vehicle: {vehicle_spec.name} | m={cfg.m} kg | D={cfg.diameter_m} m | A_ref={cfg.A_ref:.2f} m^2 | R_n={cfg.R_n} m"
    )
    results_root = ensure_results_dir(args.results_dir)
    validation_dir = os.path.join(results_root, "validation", vehicle_key)
    maxrange_dir = os.path.join(results_root, "maxrange", vehicle_key)

    seed_sigma = None
    if args.seed_bank_json:
        seed_sigma = load_warmstart(args.seed_bank_json)
        if seed_sigma is not None:
            print(f"Loaded seed bank profile from {args.seed_bank_json}")

    run_validation = args.run_validation or args.run_both
    run_maxrange = args.run_max_range or args.run_both
    if not (args.run_validation or args.run_max_range or args.run_both):
        run_validation = True
        run_maxrange = True

    val_sigma = None
    val_res = None
    if RUN_OPTIMIZATION and run_validation:
        cfg.validation_mode = True
        cfg.validation = VALIDATION_PRESETS.get(vehicle_key, cfg.validation)
        if vehicle_key != "orion":
            print(
                "WARNING: Validation targets/regression are Orion/Artemis-I-like. Non-Orion vehicle may fail regression."
            )
        val_sigma, val_res = run_mode(
            "Validation",
            cfg,
            objective_mode="validation",
            out_dir=validation_dir,
            warmstart_path=os.path.join(validation_dir, "best_sigma_knots_deg.json") if not args.no_warmstart else None,
            seed_sigma_knots_deg=seed_sigma,
            vehicle_key=vehicle_key,
            vehicle_name=vehicle_spec.name,
        )

    max_sigma = None
    max_res = None
    if RUN_OPTIMIZATION and run_maxrange:
        cfg.validation_mode = False
        seed_for_max = val_sigma if val_sigma is not None else seed_sigma
        max_sigma, max_res = run_mode(
            "Max Range",
            cfg,
            objective_mode="maxrange",
            out_dir=maxrange_dir,
            warmstart_path=os.path.join(maxrange_dir, "best_sigma_knots_deg.json") if not args.no_warmstart else None,
            seed_sigma_knots_deg=seed_for_max,
            vehicle_key=vehicle_key,
            vehicle_name=vehicle_spec.name,
        )

    if RUN_TRADE_STUDIES and max_sigma is not None and max_res is not None:
        run_trade_studies(cfg, max_sigma, max_res)

    # Combined summary
    if (val_res is not None) or (max_res is not None):
        print("\n=== MODE COMPARISON SUMMARY ===")
        if val_res is not None:
            print(
                f"Validation: range_km={val_res['final_range']/1000.0:.1f}, apogee_km={(val_res['skip_apogee']/1000.0) if val_res['skip_apogee'] is not None else None}, peak_g={val_res['peak_g']:.2f}, peak_qdot_MW={val_res['peak_qdot']/1e6:.2f}"
            )
        if max_res is not None:
            print(
                f"MaxRange:   range_km={max_res['final_range']/1000.0:.1f}, apogee_km={(max_res['skip_apogee']/1000.0) if max_res['skip_apogee'] is not None else None}, peak_g={max_res['peak_g']:.2f}, peak_qdot_MW={max_res['peak_qdot']/1e6:.2f}"
            )


if __name__ == "__main__":
    # Usage:
    #   python skip_reentry_optimizer.py           -> runs both validation and max-range modes
    #   python skip_reentry_optimizer.py --run-validation
    #   python skip_reentry_optimizer.py --run-max-range
    #   python skip_reentry_optimizer.py --run-both
    # Optional: --results-dir <path>, --seed-bank-json <file>, --no-warmstart
    main()
