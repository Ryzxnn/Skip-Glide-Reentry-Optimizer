"""
Skip reentry trajectory optimizer (3-DOF, rotating Earth, 3-phase bank schedule).

Objective (initial): maximize downrange (∫ V cosγ dt) with heating and g-load limits.
Control: three piecewise-constant bank angles (σ1, σ2, σ3) with durations t1, t2.
Target: start at (φ0=0°, θ0=0°), aim for (φt=20°N, θt=25°E) with downrange/lateral tolerances.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.optimize import minimize

# Earth parameters
R_EARTH = 6_378_137.0  # m
MU_EARTH = 3.986_004_418e14  # m^3/s^2
OMEGA_EARTH = 7.292_115_9e-5  # rad/s
G0 = 9.80665  # m/s^2

# Atmosphere
RHO0 = 1.225  # kg/m^3
H_SCALE = 7_500.0  # m

# Heating constants
K_CONV = 1.7415e-4  # Sutton-Graves constant for normalized form (W/(m^2.5·kg^0.5·s^-3))
V_CIRC = 7900.0     # Circular orbit velocity (m/s)

# Constraints (initial test values)
Q_DOT_MAX = 100e6  # W/m^2 (100 MW/m^2)
N_G_MAX = 15.0  # g
Q_TOTAL_MAX = 5e9  # J/m^2

# Skip thresholds and limits
H_EI = 120_000.0  # m
H_SKIP_OUT = 200_000.0  # m (permanent skip-out detection)
V_PARACHUTE = 200.0  # m/s
T_MAX = 5_000.0  # s


def deg2rad(deg: float) -> float:
    return math.radians(deg)


def rad2deg(rad: float) -> float:
    return math.degrees(rad)


def wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def atmosphere_density(h: float) -> float:
    """Simple exponential atmosphere."""
    if h < 0:
        h = 0.0
    return RHO0 * math.exp(-h / H_SCALE)


def gravity(r: float) -> float:
    return MU_EARTH / (r * r)


def initial_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Initial great-circle bearing from point 1 to point 2."""
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return math.atan2(x, y)


def central_angle(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle central angle between two points."""
    return math.acos(
        max(
            -1.0,
            min(1.0, math.sin(lat1) * math.sin(lat2) + math.cos(lat1) * math.cos(lat2) * math.cos(lon2 - lon1)),
        )
    )


def cross_down_errors(
    lat_start: float, lon_start: float, lat_target: float, lon_target: float, lat_final: float, lon_final: float
) -> Tuple[float, float]:
    """
    Compute downrange and crossrange errors relative to the great-circle from start to target.
    Positive downrange error means overshoot; positive crossrange is to the right of the path.
    """
    d12 = central_angle(lat_start, lon_start, lat_target, lon_target)
    d13 = central_angle(lat_start, lon_start, lat_final, lon_final)
    th12 = initial_bearing(lat_start, lon_start, lat_target, lon_target)
    th13 = initial_bearing(lat_start, lon_start, lat_final, lon_final)
    d_xt = math.asin(math.sin(d13) * math.sin(th13 - th12))
    d_at = math.atan2(math.sin(d13) * math.cos(th13 - th12), math.cos(d13))
    cross_m = d_xt * R_EARTH
    down_m = (d_at - d12) * R_EARTH
    return down_m, cross_m


@dataclass
class VehicleParams:
    m: float = 8_900.0
    S_ref: float = 19.35
    C_D: float = 1.3
    L_to_D: float = 0.35
    R_N: float = 6.25


@dataclass
class ControlSchedule:
    sigma1: float  # rad
    sigma2: float  # rad
    sigma3: float  # rad
    t1: float  # s
    t2: float  # s

    def sigma(self, t: float) -> float:
        if t <= self.t1:
            return self.sigma1
        if t <= self.t1 + self.t2:
            return self.sigma2
        return self.sigma3


@dataclass
class SimulationConfig:
    v0: float = 11_000.0  # m/s
    gamma0_deg: float = -5.5  # deg
    lat0_deg: float = 0.0
    lon0_deg: float = 0.0
    lat_t_deg: float = 20.0
    lon_t_deg: float = 25.0
    q_dot_max: float = Q_DOT_MAX
    n_g_max: float = N_G_MAX
    q_total_max: float = Q_TOTAL_MAX
    down_tol_m: float = 500_000.0
    lat_tol_m: float = 100_000.0
    h_ei: float = H_EI
    h_skip_out: float = H_SKIP_OUT
    v_parachute: float = V_PARACHUTE
    t_max: float = T_MAX
    max_step: float = 1.0
    rtol: float = 1e-6
    atol: float = 1e-8
    verbose: bool = False


@dataclass
class TrajectoryResult:
    t: np.ndarray
    y: np.ndarray
    q_dot: np.ndarray
    q_conv: np.ndarray
    q_rad: np.ndarray
    n_g: np.ndarray
    downrange_m: float
    q_dot_peak: float
    q_total: float
    n_g_peak: float
    down_err_m: float
    cross_err_m: float
    terminated_event: str
    success: bool


def build_rhs(control_fn: Callable[[float], float], vehicle: VehicleParams) -> Callable[[float, np.ndarray], List[float]]:
    def rhs(t: float, y: np.ndarray) -> List[float]:
        r, theta, phi, V, gamma, psi = y
        sigma = control_fn(t)
        h = r - R_EARTH
        rho = atmosphere_density(h)
        D = 0.5 * rho * V * V * vehicle.S_ref * vehicle.C_D
        L = vehicle.L_to_D * D
        g = gravity(r)

        sin_g = math.sin(gamma)
        cos_g = max(1e-8, abs(math.cos(gamma))) * math.copysign(1.0, math.cos(gamma))
        sin_p = math.sin(phi)
        cos_p = max(1e-8, abs(math.cos(phi))) * math.copysign(1.0, math.cos(phi))
        sin_s = math.sin(psi)
        cos_s = math.cos(psi)

        drdt = V * sin_g
        dthetadt = V * math.cos(gamma) * sin_s / (r * cos_p)
        dphidt = V * math.cos(gamma) * cos_s / r

        omega2_term = OMEGA_EARTH * OMEGA_EARTH * r * cos_p
        dVdt = -D / vehicle.m - g * sin_g + omega2_term * (sin_g * cos_p - math.cos(gamma) * sin_p * sin_s)

        dgamdt = (
            (L * math.cos(sigma)) / (vehicle.m * V)
            - (g - V * V / r) * math.cos(gamma) / V
            + 2 * OMEGA_EARTH * cos_p * cos_s
            + omega2_term * (math.cos(gamma) * cos_p + sin_g * sin_p * sin_s) / V
        )

        denom = max(1e-6, abs(math.cos(gamma))) * math.copysign(1.0, math.cos(gamma))
        dpsidt = (
            (L * math.sin(sigma)) / (vehicle.m * V * denom)
            - (V / r) * math.cos(gamma) * sin_s * math.tan(phi)
            + 2 * OMEGA_EARTH * (math.tan(gamma) * cos_p * sin_s - sin_p)
            - (OMEGA_EARTH * OMEGA_EARTH * r * sin_p * cos_p * cos_s) / (V * denom)
        )

        return [drdt, dthetadt, dphidt, dVdt, dgamdt, dpsidt]

    return rhs


def make_events(cfg: SimulationConfig) -> List[Callable[[float, np.ndarray], float]]:
    def ground_event(t: float, y: np.ndarray) -> float:
        return y[0] - R_EARTH

    ground_event.terminal = True
    ground_event.direction = -1

    def parachute_event(t: float, y: np.ndarray) -> float:
        return y[3] - cfg.v_parachute

    parachute_event.terminal = True
    parachute_event.direction = -1

    def skip_out_event(t: float, y: np.ndarray) -> float:
        return y[0] - (R_EARTH + cfg.h_skip_out)

    skip_out_event.terminal = True
    skip_out_event.direction = 1

    return [ground_event, parachute_event, skip_out_event]


def compute_heating_and_loads(vehicle: VehicleParams, V: np.ndarray, rho: np.ndarray, D: np.ndarray, L: np.ndarray):
    """Compute heating rates and g-loads using Detra-Kemp-Riddell correlation."""

    # Stagnation point convective heating (Detra-Kemp-Riddell)
    # q_conv = C * sqrt(rho / R_N) * V^3
    # where C = 1.83e-4 for SI units (kg^0.5 / m^2.5)
    C_DKR = 1.83e-4  # Detra-Kemp-Riddell constant

    q_conv = C_DKR * np.sqrt(rho / vehicle.R_N) * (V ** 3)  # W/m²

    # Radiative heating (simplified, dominant at very high velocities)
    q_rad = 5.67e-11 * rho * (V ** 3.15)  # W/m²

    q_tot = q_conv + q_rad

    # Aerodynamic acceleration
    a_aero = np.sqrt(D * D + L * L) / vehicle.m
    n_g = a_aero / G0

    if q_tot.size > 0:
        idx_peak = int(np.argmax(q_tot))
        print(f"Peak heating at idx {idx_peak}:")
        print(f"  q_conv={q_conv[idx_peak]/1e6:.2f} MW/m², q_rad={q_rad[idx_peak]/1e6:.2f} MW/m²")
        print(f"  q_tot={q_tot[idx_peak]/1e6:.2f} MW/m², V={V[idx_peak]:.0f} m/s, rho={rho[idx_peak]:.2e}")

    return q_conv, q_rad, q_tot, n_g


def simulate(
    schedule: ControlSchedule, vehicle: VehicleParams, cfg: SimulationConfig, heading0: Optional[float] = None
) -> TrajectoryResult:
    lat0 = deg2rad(cfg.lat0_deg)
    lon0 = deg2rad(cfg.lon0_deg)
    lat_t = deg2rad(cfg.lat_t_deg)
    lon_t = deg2rad(cfg.lon_t_deg)

    psi0 = heading0 if heading0 is not None else initial_bearing(lat0, lon0, lat_t, lon_t)
    y0 = np.array(
        [
            R_EARTH + cfg.h_ei,
            lon0,
            lat0,
            cfg.v0,
            deg2rad(cfg.gamma0_deg),
            psi0,
        ]
    )

    control_fn = schedule.sigma
    rhs = build_rhs(control_fn, vehicle)
    events = make_events(cfg)
    sol = solve_ivp(
        rhs,
        t_span=(0.0, cfg.t_max),
        y0=y0,
        method="RK45",
        dense_output=True,
        max_step=cfg.max_step,
        rtol=cfg.rtol,
        atol=cfg.atol,
        events=events,
    )

    if cfg.verbose:
        print("\nDiagnostics:")
        print(f"Final state: h={sol.y[0][-1]-R_EARTH:.0f} m, V={sol.y[3][-1]:.0f} m/s")
        print(f"Density at EI (120 km): {RHO0 * math.exp(-120000.0 / H_SCALE):.2e} kg/m^3")
        print(f"Density at 60 km: {RHO0 * math.exp(-60000.0 / H_SCALE):.2e} kg/m^3")
        idx = min(100, len(sol.t) - 1)
        r_sample = sol.y[0][idx]
        h_sample = r_sample - R_EARTH
        V_sample = sol.y[3][idx]
        rho_sample = atmosphere_density(h_sample)
        D_sample = 0.5 * rho_sample * V_sample * V_sample * vehicle.S_ref * vehicle.C_D
        print(f"At t={sol.t[idx]:.1f}s: h={h_sample/1000:.1f} km, V={V_sample:.0f} m/s, D={D_sample:.0f} N")

    t = sol.t
    y = sol.y
    r = y[0]
    theta = y[1]
    phi = y[2]
    V = y[3]
    gamma = y[4]

    h = r - R_EARTH
    rho = np.array([atmosphere_density(val) for val in h])
    D = 0.5 * rho * V * V * vehicle.S_ref * vehicle.C_D
    L = vehicle.L_to_D * D
    q_conv, q_rad, q_tot, n_g = compute_heating_and_loads(vehicle, V, rho, D, L)

    downrange_m = float(np.trapezoid(V * np.cos(gamma), t))
    q_peak = float(np.max(q_tot)) if q_tot.size else 0.0
    n_g_peak = float(np.max(n_g)) if n_g.size else 0.0
    q_total = float(np.trapezoid(q_tot, t)) if q_tot.size else 0.0

    lat_final = float(phi[-1])
    lon_final = float(theta[-1])
    down_err, cross_err = cross_down_errors(lat0, lon0, lat_t, lon_t, lat_final, lon_final)

    event_names = {0: "ground", 1: "parachute", 2: "skip-out"}
    term = "t_max"
    if sol.t_events:
        for idx, arr in enumerate(sol.t_events):
            if arr.size > 0:
                term = event_names.get(idx, "event")
                break

    success = sol.success and term in ("ground", "parachute")

    return TrajectoryResult(
        t=t,
        y=y,
        q_dot=q_tot,
        q_conv=q_conv,
        q_rad=q_rad,
        n_g=n_g,
        downrange_m=downrange_m,
        q_dot_peak=q_peak,
        q_total=q_total,
        n_g_peak=n_g_peak,
        down_err_m=down_err,
        cross_err_m=cross_err,
        terminated_event=term,
        success=success,
    )


def evaluate_constraints(res: TrajectoryResult, cfg: SimulationConfig) -> Dict[str, float]:
    """Return margins (positive = satisfied)."""
    return {
        "heat_peak": cfg.q_dot_max - res.q_dot_peak,
        "heat_total": cfg.q_total_max - res.q_total,
        "g_load": cfg.n_g_max - res.n_g_peak,
        "downrange_tol": cfg.down_tol_m - abs(res.down_err_m),
        "lateral_tol": cfg.lat_tol_m - abs(res.cross_err_m),
    }


def objective_factory(vehicle: VehicleParams, cfg: SimulationConfig, heading0: float):
    cache: Dict[Tuple[float, ...], TrajectoryResult] = {}

    def unpack(x: np.ndarray) -> ControlSchedule:
        return ControlSchedule(
            sigma1=x[0],
            sigma2=x[1],
            sigma3=x[2],
            t1=x[3],
            t2=x[4],
        )

    def eval_res(x: np.ndarray) -> TrajectoryResult:
        key = tuple(np.round(x, 8))
        if key not in cache:
            cache[key] = simulate(unpack(x), vehicle, cfg, heading0=heading0)
        return cache[key]

    def objective(x: np.ndarray) -> float:
        res = eval_res(x)
        margins = evaluate_constraints(res, cfg)
        penalty = 0.0
        for m in margins.values():
            if m < 0:
                penalty += 1e6 * (-m) ** 2
        if res.terminated_event == "t_max":
            penalty += 1e12
        if not res.success:
            penalty += 1e9
        if res.terminated_event == "skip-out":
            penalty += 1e8
        if res.n_g_peak > cfg.n_g_max:
            penalty += 1e8 * (res.n_g_peak - cfg.n_g_max) ** 2
        range_error = abs(res.down_err_m)
        penalty += 1e5 * range_error
        # maximize downrange -> minimize negative
        return -res.downrange_m + penalty

    constraints = [
        {"type": "ineq", "fun": lambda x, k=k: evaluate_constraints(eval_res(x), cfg)[k]}
        for k in ["heat_peak", "heat_total", "g_load", "downrange_tol", "lateral_tol"]
    ]

    bounds = [
        (deg2rad(120.0), deg2rad(180.0)),  # sigma1
        (deg2rad(-30.0), deg2rad(60.0)),  # sigma2
        (deg2rad(150.0), deg2rad(180.0)),  # sigma3
        (100.0, 300.0),  # t1
        (400.0, 1_200.0),  # t2
    ]

    return objective, constraints, bounds, eval_res


def optimize_schedule(vehicle: VehicleParams, cfg: SimulationConfig, heading0: float):
    objective, constraints, bounds, eval_res = objective_factory(vehicle, cfg, heading0)
    x0 = np.array(
        [
            deg2rad(160.0),
            deg2rad(20.0),
            deg2rad(170.0),
            180.0,
            600.0,
        ]
    )

    result = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"maxiter": 300, "ftol": 1e-5, "disp": True, "iprint": 2},
    )

    best_res = eval_res(result.x)
    best_schedule = ControlSchedule(*result.x)
    return result, best_schedule, best_res


def plot_results(res: TrajectoryResult, cfg: SimulationConfig, out_prefix: str = "skip_reentry"):
    t = res.t
    r, theta, phi, V, gamma, psi = res.y
    h = r - R_EARTH
    downrange_series = cumulative_trapezoid(V * np.cos(gamma), t, initial=0.0)
    downrange_km = downrange_series / 1e3

    plt.figure()
    plt.plot(downrange_km, h / 1e3)
    plt.xlabel("Downrange (km)")
    plt.ylabel("Altitude (km)")
    plt.title("Altitude vs Downrange")
    plt.grid(True)
    plt.savefig(f"{out_prefix}_alt_vs_downrange.png", dpi=150, bbox_inches="tight")

    plt.figure()
    plt.plot(V / 1e3, h / 1e3)
    plt.xlabel("Velocity (km/s)")
    plt.ylabel("Altitude (km)")
    plt.title("Velocity vs Altitude")
    plt.grid(True)
    plt.savefig(f"{out_prefix}_vel_vs_alt.png", dpi=150, bbox_inches="tight")

    plt.figure()
    plt.plot(t, res.q_dot / 1e6, label="Total")
    plt.plot(t, res.q_conv / 1e6, "--", label="Convective")
    plt.plot(t, res.q_rad / 1e6, "--", label="Radiative")
    plt.axhline(cfg.q_dot_max / 1e6, color="r", linestyle=":", label="q_dot_max")
    plt.xlabel("Time (s)")
    plt.ylabel("Heating Rate (MW/m^2)")
    plt.title("Heating Rate vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_prefix}_heating_vs_time.png", dpi=150, bbox_inches="tight")

    plt.figure()
    plt.plot(t, res.n_g)
    plt.axhline(cfg.n_g_max, color="r", linestyle=":", label="n_g_max")
    plt.xlabel("Time (s)")
    plt.ylabel("g-load (g)")
    plt.title("G-load vs Time")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{out_prefix}_gload_vs_time.png", dpi=150, bbox_inches="tight")

    plt.close("all")


def format_schedule(schedule: ControlSchedule) -> str:
    return (
        f"σ1={rad2deg(schedule.sigma1):.1f} deg, σ2={rad2deg(schedule.sigma2):.1f} deg, "
        f"σ3={rad2deg(schedule.sigma3):.1f} deg, t1={schedule.t1:.1f} s, t2={schedule.t2:.1f} s"
    )


def main():
    vehicle = VehicleParams()
    cfg = SimulationConfig()
    cfg.verbose = True

    lat0 = deg2rad(cfg.lat0_deg)
    lon0 = deg2rad(cfg.lon0_deg)
    lat_t = deg2rad(cfg.lat_t_deg)
    lon_t = deg2rad(cfg.lon_t_deg)
    heading0 = initial_bearing(lat0, lon0, lat_t, lon_t)

    print("Initial heading to target (deg):", rad2deg(heading0))

    opt_result, schedule, traj = optimize_schedule(vehicle, cfg, heading0)

    print("Optimization success:", opt_result.success, "status:", opt_result.status)
    print("Optimized schedule:", format_schedule(schedule))
    margins = evaluate_constraints(traj, cfg)
    print(
        f"Downrange: {traj.downrange_m/1e3:.1f} km, "
        f"q_peak: {traj.q_dot_peak/1e6:.2f} MW/m^2, "
        f"Q_total: {traj.q_total/1e9:.3f} GJ/m^2, "
        f"g_peak: {traj.n_g_peak:.2f} g"
    )
    print(
        f"Downrange error: {traj.down_err_m/1e3:.1f} km (tol {cfg.down_tol_m/1e3:.1f}), "
        f"Lateral error: {traj.cross_err_m/1e3:.1f} km (tol {cfg.lat_tol_m/1e3:.1f})"
    )
    print("Terminated by:", traj.terminated_event)
    print("Constraint margins (positive = ok):", {k: f"{v:.2e}" for k, v in margins.items()})

    plot_results(traj, cfg)
    print("Plots saved with prefix 'skip_reentry_' in current directory.")


if __name__ == "__main__":
    main()
