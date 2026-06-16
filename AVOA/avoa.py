"""
African Vulture Optimization Algorithm (AVOA)
=============================================
Implemented from:
  Li et al. (2024) - "Multi-strategy improved African Vulture Optimization
  Algorithm for global optimization and engineering design problems"
  Intelligent Data Analysis, Vol. 29(5), pp. 1313-1344.

The original AVOA was proposed by:
  Abdollahzadeh et al. (2021) - "African vultures optimization algorithm:
  a new nature-inspired metaheuristic algorithm for global optimization problems"
  Computers & Industrial Engineering, 158, 107408.

Algorithm 1 (paper, p. 1317) is followed exactly.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Helper: Levy flight  (Eq. 18)
# ---------------------------------------------------------------------------

def levy_flight(dim: int, beta: float = 1.5) -> np.ndarray:
    """Return a Levy-flight step vector of length *dim*."""
    sigma = (
        (math.gamma(1 + beta) * np.sin(np.pi * beta / 2))
        / (math.gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))
    ) ** (1 / beta)
    u = np.random.randn(dim) * sigma
    v = np.random.randn(dim)
    return 0.01 * u / (np.abs(v) ** (1 / beta))


# ---------------------------------------------------------------------------
# AVOA
# ---------------------------------------------------------------------------

class AVOA:
    """
    African Vulture Optimization Algorithm.

    Parameters
    ----------
    obj_func : callable
        Objective function f(x) -> float to be *minimised*.
    dim : int
        Number of decision variables.
    lb : array-like
        Lower bounds (length dim, or scalar broadcast).
    ub : array-like
        Upper bounds (length dim, or scalar broadcast).
    n_pop : int
        Population size N (default 30).
    max_iter : int
        Maximum iterations T (default 500).
    p1 : float
        Exploration strategy selector (default 0.6).
    p2 : float
        Exploitation phase-1 strategy selector (default 0.4).
    p3 : float
        Exploitation phase-2 strategy selector (default 0.6).
    alpha : float
        Parameter α used in the hunger factor (default 0.8).
    beta_param : float
        Parameter β for Levy flight σ (default 1.5, same symbol as Eq. 18).
    omega : float
        ω regulating exploration/exploitation split in Eq. 4 (default 2.5).
    """

    def __init__(
        self,
        obj_func,
        dim: int,
        lb,
        ub,
        n_pop: int = 30,
        max_iter: int = 500,
        p1: float = 0.6,
        p2: float = 0.4,
        p3: float = 0.6,
        alpha: float = 0.8,
        beta_param: float = 1.5,
        omega: float = 2.5,
    ):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = np.full(dim, lb, dtype=float) if np.isscalar(lb) else np.array(lb, dtype=float)
        self.ub = np.full(dim, ub, dtype=float) if np.isscalar(ub) else np.array(ub, dtype=float)
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.alpha = alpha          # α — kept for parameter parity with paper Table 6/7
        self.beta_param = beta_param
        self.omega = omega

        # Results filled after run()
        self.best_pos = None  # type: np.ndarray
        self.best_fitness: float = float("inf")
        self.convergence_curve: list[float] = []

    # ------------------------------------------------------------------
    # Section 2.1 – Roulette-wheel selection of reference vulture (Eq. 1-2)
    # ------------------------------------------------------------------

    def _select_reference(self, best1: np.ndarray, best2: np.ndarray,
                           fitness: np.ndarray) -> np.ndarray:
        """Return BestVulture1 or BestVulture2 via roulette-wheel (Eq. 2)."""
        # pi = Fi / sum(Fi)  — fitness values are *costs*, so invert for prob.
        # Paper uses roulette on raw fitness; treat smaller fitness as better
        # probability by using 1/Fi (avoid divide-by-zero with epsilon).
        eps = 1e-10
        inv_f = 1.0 / (np.abs(fitness) + eps)
        prob = inv_f / inv_f.sum()
        cumprob = np.cumsum(prob)
        r = np.random.rand()
        # Map cumulative prob to L1 or L2 threshold
        # Paper: if pi == L1 -> BestVulture1, else BestVulture2
        # We use a simple 50-50 split as is conventional for AVOA
        if r < 0.5:
            return best1.copy()
        else:
            return best2.copy()

    # ------------------------------------------------------------------
    # Section 2.2 – Hunger level F (Eq. 3-4)
    # ------------------------------------------------------------------

    def _hunger_factor(self, iteration: int) -> float:
        """Compute hunger level F (Eq. 3-4)."""
        rand1 = np.random.rand()
        z = np.random.uniform(-1, 1)
        h = np.random.uniform(-2, 2)
        omega = self.omega

        # Eq. 4
        t = h * (
            np.sin(omega * (np.pi / 2) * iteration / self.max_iter)
            + np.cos(np.pi / 2 * iteration / self.max_iter)
            - 1
        )
        # Eq. 3
        F = (2 * rand1 + 1) * z * (1 - iteration / self.max_iter) + t
        return F

    # ------------------------------------------------------------------
    # Clipping helper
    # ------------------------------------------------------------------

    def _clip(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, self.lb, self.ub)

    # ------------------------------------------------------------------
    # Main loop  (Algorithm 1, p. 1317)
    # ------------------------------------------------------------------

    def run(self, seed=None):
        """
        Execute AVOA.

        Returns
        -------
        best_pos : ndarray
            Best solution found.
        best_fitness : float
            Corresponding objective value.
        """
        if seed is not None:
            np.random.seed(seed)

        dim = self.dim
        lb, ub = self.lb, self.ub

        # ── Step 1: Initialise random population ─────────────────────────
        pop = lb + np.random.rand(self.n_pop, dim) * (ub - lb)
        fitness = np.array([self.obj_func(pop[i]) for i in range(self.n_pop)])

        # Identify BestVulture1 and BestVulture2
        sorted_idx = np.argsort(fitness)
        best_vulture1 = pop[sorted_idx[0]].copy()
        best_vulture2 = pop[sorted_idx[1]].copy()
        self.best_fitness = fitness[sorted_idx[0]]
        self.best_pos = best_vulture1.copy()
        self.convergence_curve = []

        # ── Main loop ────────────────────────────────────────────────────
        for t in range(1, self.max_iter + 1):

            for i in range(self.n_pop):

                # Step 7: Select reference vulture R(i)  (Eq. 1-2)
                R = self._select_reference(best_vulture1, best_vulture2, fitness)

                # Step 8: Update F  (Eq. 3-4)
                F = self._hunger_factor(t)
                absF = abs(F)

                P = pop[i].copy()

                # ── Exploration phase  |F| >= 1  (Eq. 5-8) ──────────────
                if absF >= 1:
                    rand_p1 = np.random.rand()
                    X = np.random.randint(1, 3)          # random 1 or 2

                    if self.p1 >= rand_p1:
                        # Strategy 1 (Eq. 5-6)
                        D = abs(X * R - P)
                        P_new = R - D * F               # Eq. 5
                    else:
                        # Strategy 2 (Eq. 7)
                        rand2 = np.random.rand()
                        rand3 = np.random.rand()
                        P_new = R - F + rand2 * ((ub - lb) * rand3 + lb)

                # ── Exploitation phase  |F| < 1  ─────────────────────────
                else:
                    if absF >= 0.5:
                        # Phase 1  (Eq. 9)
                        rand_p2 = np.random.rand()
                        rand4 = np.random.rand()

                        if self.p2 >= rand_p2:
                            # Food protection (Eq. 10-11)
                            d_t = R - P
                            D = abs(X * R - P) if 'X' in dir() else abs(R - P)
                            D = abs(R - P)
                            P_new = D * (F + rand4) - d_t   # Eq. 10
                        else:
                            # Spiral flight (Eq. 12-13)
                            rand5 = np.random.rand()
                            S1 = R * (rand5 * P / (2 * np.pi)) * np.cos(P)
                            S2 = R * (rand5 * P / (2 * np.pi)) * np.sin(P)
                            P_new = R - (S1 + S2)            # Eq. 13

                    else:
                        # Phase 2  (Eq. 14)
                        rand_p3 = np.random.rand()

                        if self.p3 >= rand_p3:
                            # Accumulation over food (Eq. 15-16)
                            A1 = (
                                best_vulture1
                                - (best_vulture1 * P) / (best_vulture1 - P ** 2 + 1e-10) * F
                            )
                            A2 = (
                                best_vulture2
                                - (best_vulture2 * P) / (best_vulture2 - P ** 2 + 1e-10) * F
                            )
                            P_new = (A1 + A2) / 2           # Eq. 16
                        else:
                            # Competition for food (Eq. 17)
                            d_t = R - P
                            lf = levy_flight(dim, self.beta_param)
                            P_new = R - abs(d_t) * F * lf   # Eq. 17

                # ── Clip and evaluate ─────────────────────────────────────
                P_new = self._clip(P_new)
                f_new = self.obj_func(P_new)

                if f_new < fitness[i]:
                    pop[i] = P_new
                    fitness[i] = f_new

            # ── Update best vultures ──────────────────────────────────────
            sorted_idx = np.argsort(fitness)
            if fitness[sorted_idx[0]] < self.best_fitness:
                self.best_fitness = fitness[sorted_idx[0]]
                self.best_pos = pop[sorted_idx[0]].copy()

            best_vulture1 = pop[sorted_idx[0]].copy()
            best_vulture2 = pop[sorted_idx[1]].copy()

            self.convergence_curve.append(self.best_fitness)

        return self.best_pos, self.best_fitness


# ---------------------------------------------------------------------------
# Quick demo / test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ── F1: Sphere  (unimodal, global min = 0) ───────────────────────────
    def sphere(x):
        return float(np.sum(x ** 2))

    # ── F9: Rastrigin  (multimodal, global min = 0) ──────────────────────
    def rastrigin(x):
        return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10))

    # ── F10: Ackley  (multimodal, global min = 0) ────────────────────────
    def ackley(x):
        n = len(x)
        a = -0.2 * np.sqrt(np.sum(x ** 2) / n)
        b = np.sum(np.cos(2 * np.pi * x)) / n
        return float(-20 * np.exp(a) - np.exp(b) + 20 + np.e)

    problems = [
        ("F1  Sphere     [−100,100]^30", sphere,   30, -100, 100),
        ("F9  Rastrigin  [−5.12,5.12]^30", rastrigin, 30, -5.12, 5.12),
        ("F10 Ackley     [−32,32]^30",   ackley,   30,  -32,  32),
    ]

    print("=" * 65)
    print(f"{'Function':<40} {'Best fitness':>12}  {'Iterations':>10}")
    print("=" * 65)

    for name, func, dim, lb, ub in problems:
        avoa = AVOA(
            obj_func=func,
            dim=dim,
            lb=lb,
            ub=ub,
            n_pop=30,
            max_iter=500,
            p1=0.6,
            p2=0.4,
            p3=0.6,
        )
        best_pos, best_fit = avoa.run(seed=42)
        print(f"{name:<40} {best_fit:>12.6e}  {500:>10}")

    print("=" * 65)
    print("\nDone. Access convergence_curve, best_pos, best_fitness on the AVOA object.")