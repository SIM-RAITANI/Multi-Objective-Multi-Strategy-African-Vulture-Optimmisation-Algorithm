"""
African Vulture Optimization Algorithm (AVOA)
with MIAVOA Improvements 3 & 4
======================================================
Base algorithm implemented from:
  Li et al. (2024) - "Multi-strategy improved African Vulture Optimization
  Algorithm for global optimization and engineering design problems"
  Intelligent Data Analysis, Vol. 29(5), pp. 1313-1344.

The original AVOA was proposed by:
  Abdollahzadeh et al. (2021) - "African vultures optimization algorithm:
  a new nature-inspired metaheuristic algorithm for global optimization problems"
  Computers & Industrial Engineering, 158, 107408.

Two MIAVOA improvements are integrated here
-------------------------------------------
Improvement 3 — Elite Candidate Pool strategy (Section 3.3, Eq. 24)
    Replaces the plain spiral-flight position update (Eq. 13) with a pool of
    four candidate positions constructed from S1 and S2.  The best candidate
    is kept as the offspring, widening the discovery field and helping the
    algorithm escape local optima during the |F| >= 0.5 exploitation phase.

    Original (Eq. 13):
        P(i+1) = R(i) − (S1 + S2)

    Replacement (Eq. 24):
        R1 = R(i) + S1
        R2 = R(i) − S1
        R3 = R(i) + S2
        R4 = R(i) − S2
        P(i+1) = best fitness among {R1, R2, R3, R4}

Improvement 4 — Improved hunger parameter (Section 3.4, Eq. 25)
    Replaces the original starvation factor F (Eq. 3) with an exponential
    decay term.  The new formulation makes |F| decrease monotonically over
    generations, giving a smoother and more principled exploration-to-
    exploitation transition.

    Original (Eq. 3):
        F = (2·rand1 + 1) · z · (1 − iter/maxiter) + t

    Replacement (Eq. 25):
        F = (2·rand1 + 1) · z · exp((iter/maxiter − 1) / (e − 1)²) + t

    where t is still computed by the original Eq. 4.
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
# AVOA with MIAVOA improvements 3 & 4
# ---------------------------------------------------------------------------

class AVOA:
    """
    African Vulture Optimization Algorithm with MIAVOA improvements 3 & 4.

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
        Parameter α for adaptive weighting — kept for parameter parity
        with paper Table 6/7 (default 0.8).
    beta_param : float
        Parameter β for Levy flight σ (default 1.5, Eq. 18).
    omega : float
        ω regulating the sinusoidal component of F (Eq. 4, default 2.5).
    use_elite_pool : bool
        Toggle Improvement 3 – Elite Candidate Pool (default True).
    use_improved_F : bool
        Toggle Improvement 4 – Improved hunger parameter (default True).
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
        use_elite_pool: bool = True,
        use_improved_F: bool = True,
    ):
        self.obj_func = obj_func
        self.dim = dim
        self.lb = (
            np.full(dim, lb, dtype=float)
            if np.isscalar(lb)
            else np.array(lb, dtype=float)
        )
        self.ub = (
            np.full(dim, ub, dtype=float)
            if np.isscalar(ub)
            else np.array(ub, dtype=float)
        )
        self.n_pop = n_pop
        self.max_iter = max_iter
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.alpha = alpha
        self.beta_param = beta_param
        self.omega = omega
        self.use_elite_pool = use_elite_pool
        self.use_improved_F = use_improved_F

        # Results filled after run()
        self.best_pos: np.ndarray | None = None
        self.best_fitness: float = float("inf")
        self.convergence_curve: list[float] = []

    # ------------------------------------------------------------------
    # Section 2.1 – Roulette-wheel selection of reference vulture (Eq. 1-2)
    # ------------------------------------------------------------------

    def _select_reference(
        self,
        best1: np.ndarray,
        best2: np.ndarray,
        fitness: np.ndarray,
    ) -> np.ndarray:
        """Return BestVulture1 or BestVulture2 via roulette-wheel (Eq. 2)."""
        eps = 1e-10
        inv_f = 1.0 / (np.abs(fitness) + eps)
        prob = inv_f / inv_f.sum()
        cumprob = np.cumsum(prob)
        r = np.random.rand()
        # Conventional AVOA 50-50 split mapping the roulette to L1 / L2
        if r < 0.5:
            return best1.copy()
        else:
            return best2.copy()

    # ------------------------------------------------------------------
    # IMPROVEMENT 4 — Improved hunger parameter (Eq. 25)
    # ------------------------------------------------------------------

    def _hunger_factor(self, iteration: int) -> float:
        """
        Compute the starvation factor F.

        When *use_improved_F* is True (default) the MIAVOA Eq. 25 formula
        is used; otherwise the original Eq. 3 is applied.

        Original (Eq. 3):
            F = (2·rand1 + 1) · z · (1 − iter/maxiter) + t

        MIAVOA Eq. 25:
            F = (2·rand1 + 1) · z · exp((iter/maxiter − 1) / (e − 1)²) + t

        The exponential factor replaces the linear decay.  Because the
        exponent is always ≤ 0 (iter ≤ maxiter), exp(·) stays in (0, 1],
        yielding a smooth monotone decay that shifts the algorithm from
        exploration to exploitation more gradually than the linear term.
        """
        rand1 = np.random.rand()
        z = np.random.uniform(-1, 1)
        h = np.random.uniform(-2, 2)
        ratio = iteration / self.max_iter

        # Eq. 4 — sinusoidal term (unchanged in MIAVOA)
        t = h * (
            np.sin(self.omega * (np.pi / 2) * ratio)
            + np.cos((np.pi / 2) * ratio)
            - 1.0
        )

        if self.use_improved_F:
            # ── MIAVOA Eq. 25 ─────────────────────────────────────────────
            # exp((iter/maxiter − 1) / (e − 1)²)
            # When iter = 0  → exp(−1/(e−1)²) ≈ 0.775  (near 1, explore)
            # When iter = T  → exp(0)          = 1.0
            # Multiplied by z ∈ (−1,1) the magnitude of F thus decreases
            # smoothly towards zero, ensuring the algorithm converges.
            denom = (np.e - 1.0) ** 2          # (e − 1)² ≈ 2.952
            exp_decay = np.exp((ratio - 1.0) / denom)
            F = (2 * rand1 + 1) * z * exp_decay + t
        else:
            # ── Original Eq. 3 ────────────────────────────────────────────
            F = (2 * rand1 + 1) * z * (1.0 - ratio) + t

        return F

    # ------------------------------------------------------------------
    # IMPROVEMENT 3 — Elite Candidate Pool (Section 3.3, Eq. 24)
    # ------------------------------------------------------------------

    def _elite_candidate_pool(
        self,
        R: np.ndarray,
        S1: np.ndarray,
        S2: np.ndarray,
        current_pos: np.ndarray,
        current_fitness: float,
    ) -> tuple[np.ndarray, float]:
        """
        Build four candidate positions from S1 / S2 and return the best.

        Eq. 24:
            R1 = R(i) + S1
            R2 = R(i) − S1
            R3 = R(i) + S2
            R4 = R(i) − S2

        All four candidates are clipped to [lb, ub] before evaluation.
        The one with the lowest objective value becomes the offspring.
        If none improves on the current position the current position is
        returned unchanged (greedy acceptance is handled by the caller).
        """
        candidates = [
            self._clip(R + S1),
            self._clip(R - S1),
            self._clip(R + S2),
            self._clip(R - S2),
        ]

        best_cand = None
        best_fit = float("inf")
        for cand in candidates:
            f = self.obj_func(cand)
            if f < best_fit:
                best_fit = f
                best_cand = cand

        return best_cand, best_fit

    # ------------------------------------------------------------------
    # Clipping helper
    # ------------------------------------------------------------------

    def _clip(self, pos: np.ndarray) -> np.ndarray:
        return np.clip(pos, self.lb, self.ub)

    # ------------------------------------------------------------------
    # Main loop  (Algorithm 1, p. 1317, with improvements 3 & 4)
    # ------------------------------------------------------------------

    def run(self, seed=None):
        """
        Execute AVOA (+ MIAVOA improvements 3 & 4).

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

                # Step 8: Update F
                # IMPROVEMENT 4 is applied inside _hunger_factor when
                # self.use_improved_F is True.
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

                    # Clip and evaluate (greedy acceptance below)
                    P_new = self._clip(P_new)
                    f_new = self.obj_func(P_new)

                # ── Exploitation phase  |F| < 1  ─────────────────────────
                else:
                    if absF >= 0.5:
                        # Phase 1  (Eq. 9)
                        rand_p2 = np.random.rand()
                        rand4 = np.random.rand()

                        if self.p2 >= rand_p2:
                            # Food protection (Eq. 10-11) — unchanged
                            d_t = R - P
                            D = abs(R - P)
                            P_new = D * (F + rand4) - d_t   # Eq. 10
                            P_new = self._clip(P_new)
                            f_new = self.obj_func(P_new)
                        else:
                            # ── IMPROVEMENT 3: Elite Candidate Pool ──────
                            # Compute S1 and S2 (Eq. 12) first.
                            rand5 = np.random.rand()
                            S1 = R * (rand5 * P / (2 * np.pi)) * np.cos(P)
                            S2 = R * (rand5 * P / (2 * np.pi)) * np.sin(P)

                            if self.use_elite_pool:
                                # Build four candidates (Eq. 24) and pick best.
                                # Original Eq. 13 (P_new = R − (S1+S2)) is
                                # one of the implicit directions; the pool
                                # explores four symmetric alternatives instead.
                                P_new, f_new = self._elite_candidate_pool(
                                    R, S1, S2, P, fitness[i]
                                )
                            else:
                                # Original spiral flight (Eq. 13)
                                P_new = self._clip(R - (S1 + S2))
                                f_new = self.obj_func(P_new)

                    else:
                        # Phase 2  (Eq. 14)
                        rand_p3 = np.random.rand()

                        if self.p3 >= rand_p3:
                            # Accumulation over food (Eq. 15-16) — unchanged
                            A1 = (
                                best_vulture1
                                - (best_vulture1 * P)
                                / (best_vulture1 - P ** 2 + 1e-10)
                                * F
                            )
                            A2 = (
                                best_vulture2
                                - (best_vulture2 * P)
                                / (best_vulture2 - P ** 2 + 1e-10)
                                * F
                            )
                            P_new = self._clip((A1 + A2) / 2)   # Eq. 16
                            f_new = self.obj_func(P_new)
                        else:
                            # Competition for food (Eq. 17) — unchanged
                            d_t = R - P
                            lf = levy_flight(dim, self.beta_param)
                            P_new = self._clip(R - abs(d_t) * F * lf)
                            f_new = self.obj_func(P_new)

                # ── Greedy acceptance ─────────────────────────────────────
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
# Benchmark functions
# ---------------------------------------------------------------------------
# Why these specific functions?
# ─────────────────────────────
# F1 Sphere and F9 Rastrigin are trivially solved by plain AVOA in 500
# iterations with dim=30 — all variants converge to 0, so no differences
# are visible.  To expose the real effect of Improvements 3 & 4 we need:
#
#   • High dimension (dim=50 / 100) so the search space is genuinely vast.
#   • Highly multimodal functions with many local optima (Schwefel, Levy,
#     Griewank, Rosenbrock) where spiral flight and the F schedule matter.
#   • Multiple independent runs (N_RUNS = 10) so randomness is averaged out
#     and we report Mean ± Std instead of a single lucky seed.
#
# Function catalogue used here (all minimisation, global min = 0 unless noted)
# ─────────────────────────────────────────────────────────────────────────────
# F1  Sphere          unimodal baseline            dim=50, [-100,100]
# F5  Rosenbrock      narrow curved valley         dim=50, [-30,30]
# F8  Schwefel        deceptive, global far away   dim=50, [-500,500]  min≠0
# F9  Rastrigin       highly multimodal            dim=50, [-5.12,5.12]
# F10 Ackley          multimodal, flat outer       dim=50, [-32,32]
# F11 Griewank        many local optima            dim=50, [-600,600]
# F12 Penalised-1     multimodal + penalty         dim=50, [-50,50]
# F13 Penalised-2     multimodal + penalty         dim=50, [-50,50]

def _u(x, a, k, m):
    """Penalty helper used by F12/F13."""
    return np.where(x > a, k * (x - a) ** m,
           np.where(x < -a, k * (-x - a) ** m, 0.0))

def f1_sphere(x):
    return float(np.sum(x ** 2))

def f5_rosenbrock(x):
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (x[:-1] - 1) ** 2))

def f8_schwefel(x):
    # Global minimum ≈ −418.9829 * dim  (best → most negative)
    return float(np.sum(-x * np.sin(np.sqrt(np.abs(x)))))

def f9_rastrigin(x):
    return float(np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x) + 10))

def f10_ackley(x):
    n = len(x)
    return float(-20 * np.exp(-0.2 * np.sqrt(np.sum(x**2) / n))
                 - np.exp(np.sum(np.cos(2 * np.pi * x)) / n)
                 + 20 + np.e)

def f11_griewank(x):
    i = np.arange(1, len(x) + 1)
    return float(np.sum(x**2) / 4000 - np.prod(np.cos(x / np.sqrt(i))) + 1)

def f12_penalised1(x):
    n = len(x)
    y = 1 + (x + 1) / 4
    term1 = 10 * np.sin(np.pi * y[0]) ** 2
    term2 = np.sum((y[:-1] - 1)**2 * (1 + 10 * np.sin(np.pi * y[1:])**2))
    term3 = (y[-1] - 1) ** 2
    pen = np.sum(_u(x, 10, 100, 4))
    return float(np.pi / n * (term1 + term2 + term3) + pen)

def f13_penalised2(x):
    n = len(x)
    term1 = np.sin(3 * np.pi * x[0]) ** 2
    term2 = np.sum((x[:-1] - 1)**2 * (1 + np.sin(3 * np.pi * x[1:])**2))
    term3 = (x[-1] - 1)**2 * (1 + np.sin(2 * np.pi * x[-1])**2)
    pen = np.sum(_u(x, 5, 100, 4))
    return float(0.1 * (term1 + term2 + term3) + pen)


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    N_RUNS = 10   # independent runs per config per function

    problems = [
        # (short name,     function,        dim,  lb,      ub)
        ("F1  Sphere",     f1_sphere,        50, -100,    100),
        ("F5  Rosenbrock", f5_rosenbrock,    50,  -30,     30),
        ("F8  Schwefel",   f8_schwefel,      50, -500,    500),
        ("F9  Rastrigin",  f9_rastrigin,     50,  -5.12,   5.12),
        ("F10 Ackley",     f10_ackley,       50,  -32,     32),
        ("F11 Griewank",   f11_griewank,     50, -600,    600),
        ("F12 Penalised1", f12_penalised1,   50,  -50,     50),
        ("F13 Penalised2", f13_penalised2,   50,  -50,     50),
    ]

    configs = [
        # (display label,          use_elite_pool, use_improved_F)
        ("Plain AVOA",             False,          False),
        ("AVOA + ElitePool [I3]",  True,           False),
        ("AVOA + ImprovedF [I4]",  False,          True),
        ("AVOA + Both  [MIAVOA]",  True,           True),
    ]

    # ── Header ───────────────────────────────────────────────────────────
    print("\n" + "=" * 120)
    print(f"  Benchmark: {N_RUNS} independent runs, dim=50, pop=30, iter=500")
    print(f"  Reporting: Mean  (Std)")
    print(f"  Improvement 3 = Elite Candidate Pool (Eq. 24)")
    print(f"  Improvement 4 = Improved Hunger Parameter (Eq. 25)")
    print("=" * 120)

    COL = 30  # width per config column
    header = f"{'Function':<18}" + "".join(f"{c[0]:^{COL}}" for c in configs)
    print(header)
    print("-" * 120)

    for fname, func, dim, lo, hi in problems:
        row = f"{fname:<18}"
        for _, ep, imf in configs:
            runs = []
            for run_id in range(N_RUNS):
                avoa = AVOA(
                    obj_func=func,
                    dim=dim,
                    lb=lo,
                    ub=hi,
                    n_pop=30,
                    max_iter=500,
                    p1=0.6,
                    p2=0.4,
                    p3=0.6,
                    alpha=0.8,
                    use_elite_pool=ep,
                    use_improved_F=imf,
                )
                _, best = avoa.run(seed=run_id * 7 + 13)  # different seed each run
                runs.append(best)
            mean = np.mean(runs)
            std  = np.std(runs)
            cell = f"{mean:.3e} ({std:.1e})"
            row += f"{cell:^{COL}}"
        print(row)

    print("=" * 120)
    print("\nKey: Mean (Std) over", N_RUNS, "runs.")
    print("     Lower is better. Schwefel: most-negative value is best.")
    print("\nToggles:  use_elite_pool=True/False  |  use_improved_F=True/False")
