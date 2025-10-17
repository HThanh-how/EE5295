"""
PSO optimization for VCO sizing (standalone script)
---------------------------------------------------
This script mirrors the PSO logic used in the notebook, but is packaged as a
separate .py with detailed comments for submission.

Usage (no simulator required):
    python pso_opt.py

Outputs:
    - Prints best parameters (Wn, Wp, L, C) and best cost
    - Saves convergence plot to ../figures/pso_convergence.png (if matplotlib
      is available)

Notes:
    - The cost function is behavioral (no SPICE call):
        cost = a*Power + b*(f0 - f_target)^2 + c*(PN_proxy)
      with technology bounds enforced as hard constraints.
    - This is intended to demonstrate the AI/meta-heuristic integration; the
      result is illustrative and portable.
"""

from __future__ import annotations

import math
import os
from typing import List, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt  # optional
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


class VCOOptimizer:
    """Particle Swarm Optimization for a simple VCO sizing model.

    Decision variables (per particle):
      - Wn (m), Wp (m), L (m), C (F)

    Bounds (technology-inspired):
      - Wn \in [0.5e-6, 10e-6]
      - Wp \in [1e-6, 20e-6]
      - L  \in [90e-9, 500e-9]
      - C  \in [1e-15, 100e-15]

    Cost function components:
      - Power ~ Vdd * (Wn + Wp) * k  (proxy only)
      - Frequency target term (f0 - f_target)^2 where
          f0 = 1 / (2*pi*sqrt(L*C)) * f_scale
      - Phase-noise proxy via Leeson-like Q scaling
    """

    def __init__(self) -> None:
        self.best_params: List[float] | None = None
        self.best_cost: float = float("inf")
        self.history: List[float] = []

    def vco_cost_function(self, params: List[float]) -> float:
        Wn, Wp, L, C = params

        # Hard constraints: if out of tech range, penalize heavily
        if not (0.5e-6 <= Wn <= 10e-6):
            return 1e9
        if not (1e-6 <= Wp <= 20e-6):
            return 1e9
        if not (90e-9 <= L <= 500e-9):
            return 1e9
        if not (1e-15 <= C <= 100e-15):
            return 1e9

        # Behavioral model (proxies)
        f_scale = 1e9  # scale factor to keep frequencies in practical range
        f0 = f_scale / (2.0 * math.pi * math.sqrt(L * C))

        Vdd = 1.8
        k_power = 1e-9  # arbitrary scale for power proxy
        power_proxy = Vdd * (Wn + Wp) * k_power

        # Leeson-like PN proxy with a fixed Q to reward higher f0
        Q = 20.0
        pn_proxy = -80.0 - 20.0 * math.log10(Q) - 20.0 * math.log10(max(f0, 1.0) / 1e6)

        f_target = 100e6

        # Weighted cost (tune weights as desired)
        a = 1.0
        b = 1.0 / (1e12)
        c = 1.0
        cost = a * power_proxy + b * (f0 - f_target) ** 2 + c * (pn_proxy + 100.0) ** 2
        return float(cost)

    def pso_optimize(self, n_particles: int = 30, n_iterations: int = 100,
                      seed: int | None = 42) -> Tuple[List[float], float]:
        if seed is not None:
            np.random.seed(seed)

        bounds = [
            (0.5e-6, 10e-6),   # Wn
            (1e-6, 20e-6),     # Wp
            (90e-9, 500e-9),   # L
            (1e-15, 100e-15),  # C
        ]

        # Initialize swarm
        particles: List[List[float]] = []
        velocities: List[List[float]] = []
        pbest: List[List[float]] = []
        pbest_cost: List[float] = []

        for _ in range(n_particles):
            x = [np.random.uniform(lo, hi) for lo, hi in bounds]
            v = [np.random.uniform(-0.1 * (hi - lo), 0.1 * (hi - lo)) for lo, hi in bounds]
            particles.append(x)
            velocities.append(v)
            cost = self.vco_cost_function(x)
            pbest.append(list(x))
            pbest_cost.append(cost)
            if cost < self.best_cost:
                self.best_cost = cost
                self.best_params = list(x)

        # PSO hyper-parameters
        w = 0.9
        c1 = 2.0
        c2 = 2.0

        for _ in range(n_iterations):
            for i in range(n_particles):
                # Update velocity
                for j, (lo, hi) in enumerate(bounds):
                    r1, r2 = np.random.random(2)
                    velocities[i][j] = (
                        w * velocities[i][j]
                        + c1 * r1 * (pbest[i][j] - particles[i][j])
                        + c2 * r2 * (self.best_params[j] - particles[i][j])  # type: ignore[index]
                    )
                # Update position + clamp
                for j, (lo, hi) in enumerate(bounds):
                    particles[i][j] += velocities[i][j]
                    particles[i][j] = float(np.clip(particles[i][j], lo, hi))

                # Evaluate
                cost = self.vco_cost_function(particles[i])
                if cost < pbest_cost[i]:
                    pbest[i] = list(particles[i])
                    pbest_cost[i] = cost
                    if cost < self.best_cost:
                        self.best_cost = cost
                        self.best_params = list(particles[i])

            self.history.append(self.best_cost)

        return list(self.best_params), float(self.best_cost)  # type: ignore[arg-type]


def main() -> None:
    opt = VCOOptimizer()
    best_params, best_cost = opt.pso_optimize(n_particles=30, n_iterations=100)
    Wn, Wp, L, C = best_params
    print("PSO best:")
    print(f"  Wn = {Wn*1e6:.2f} um, Wp = {Wp*1e6:.2f} um, L = {L*1e9:.0f} nm, C = {C*1e15:.1f} fF")
    print(f"  cost = {best_cost:.3e}")

    if _HAS_MPL:
        os.makedirs(os.path.join("..", "figures"), exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.semilogy(opt.history)
        plt.xlabel("Iteration")
        plt.ylabel("Best cost")
        plt.title("PSO convergence")
        out = os.path.join("..", "figures", "pso_convergence.png")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()


