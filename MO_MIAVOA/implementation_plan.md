# MO-MIAVOA Phase 1 — Architecture & Implementation Plan

## 1. Project Structure

```
d:\Projects\MO_MIAVOA\
├── config.py                  # All tunable parameters with defaults
├── main.py                    # Entry point — run optimization + visualize
│
├── environment/
│   ├── __init__.py
│   ├── models.py              # Dataclasses: Building, WindZone, Environment, Path
│   ├── generator.py           # Random environment generation (seeded)
│   ├── collision.py           # Segment-polygon intersection, min clearance
│   └── wind.py                # Wind zone exposure (per-segment clipping)
│
├── objectives/
│   ├── __init__.py
│   ├── time_obj.py            # Objective 1: Delivery Time
│   ├── energy_obj.py          # Objective 2: Energy Expenditure
│   ├── risk_obj.py            # Objective 3: Medical Payload Risk
│   └── evaluator.py           # Combined evaluator — computes (T, E, R) tuple
│
├── optimizer/
│   ├── __init__.py
│   ├── archive.py             # Pareto archive: dominance, crowding distance, insert/prune
│   ├── leader_selection.py    # Crowding-distance tournament → leader pool
│   ├── miavoa_core.py         # Base MIAVOA equations (Eq 1–25 from paper)
│   ├── crossover.py           # Splice crossover + joint repair
│   ├── mutation.py            # Gaussian jitter + bounded detour
│   └── mo_miavoa.py           # Main loop orchestrator (Section 9 of design doc)
│
├── selection/
│   ├── __init__.py
│   └── final_selection.py     # Weighted aggregation + knee point
│
├── visualization/
│   ├── __init__.py
│   └── plotting.py            # Environment plot, path overlay, Pareto front plot
│
└── tests/
    ├── test_environment.py
    ├── test_objectives.py
    ├── test_archive.py
    └── test_optimizer.py
```

---

## 2. Config Defaults (`config.py`)

All values derived from the base paper (Li et al. 2024) and design doc recommendations.

```python
# === Population & Iterations ===
POPULATION_SIZE = 30          # N — base paper uses 30
MAX_ITERATIONS = 500          # T — base paper uses 500
NUM_WAYPOINTS = 8             # n — SES recommendation

# === World ===
WORLD_WIDTH = 1000.0          # meters
WORLD_HEIGHT = 1000.0         # meters

# === MIAVOA Base Parameters (from paper Table 6) ===
P1 = 0.6                     # Exploration strategy threshold
P2 = 0.4                     # Exploitation sub-phase threshold
P3 = 0.6                     # Late exploitation threshold
ALPHA = 0.8                  # Adaptive control parameter (α)
BETA_LEVY = 1.5              # Lévy flight β
GAMMA_INIT = 2.5             # Gaussian chaos mapping γ seed

# === Archive ===
ARCHIVE_CAPACITY = 75         # C_max — midpoint of 50–100 range
LEADER_POOL_SIZE = 4          # midpoint of 3–5

# === Objective Function Coefficients ===
DRONE_CRUISE_SPEED = 15.0     # v (m/s) — typical delivery drone
K1_ENERGY_DISTANCE = 1.0     # k1 — distance weight for Energy
K2_ENERGY_WIND = 0.5          # k2 — wind weight for Energy

# Wind zone intensities
WIND_INTENSITY = {"green": 1.0, "yellow": 3.0, "red": 6.0}

# Risk sub-component weights
W_CLEARANCE = 0.4             # w_c
W_TURNING = 0.3               # w_t
W_WIND_RISK = 0.3             # w_w
CLEARANCE_DECAY_ALPHA = 0.1   # α in f(c) = exp(-α·c)
SAFETY_BUFFER = 15.0          # meters — midpoint of 10–20

# === GA Component ===
CROSSOVER_INTERVAL = 12       # iterations between crossover events
MUTATION_RATE = 0.1           # fraction of population mutated per iteration
MUTATION_SIGMA = 20.0         # Gaussian jitter std dev (meters)
REPAIR_MAX_RETRIES = 3        # crossover repair attempts
PUSHOUT_MAX_ITERS = 5         # push-out repair iterations
REPAIR_CLEARANCE = 5.0        # safety margin for tangent snap (meters)

# === Environment Generation ===
NUM_BUILDINGS = 15            # default building count
BUILDING_WIDTH_RANGE = (30, 100)
BUILDING_HEIGHT_RANGE = (30, 150)
NUM_WIND_ZONES = 4
WIND_ZONE_SIZE_RANGE = (100, 300)
SOURCE_DEST_MIN_SEPARATION = 500.0  # meters

# === Final Selection ===
DEFAULT_WEIGHTS = {"time": 0.4, "energy": 0.3, "risk": 0.3}
```

---

## 3. Base MIAVOA Equations → Code Mapping

These are the equations from Li et al. 2024 that go into `miavoa_core.py`:

### 3.1 Leader Selection (Eq 1–2)
```
R(i) = BestVulture1 if p_i ≥ L1, else BestVulture2
p_i = F_i / Σ F_i  (roulette wheel)
```
**In MO-MIAVOA**: BestVulture1/2 are replaced by leaders drawn from the crowding-distance tournament pool (design doc Section 7).

### 3.2 Hunger Factor (Eq 3–4, improved as Eq 25)
```
# Original (Eq 3-4):
t = h × (sin(ω) × |cos(ω)| - cos(ω) × |sin(ω)|)
    where ω = (π/2) × (iteration_i / max_iterations)
F = (2×rand1 + 1) × z × (1 - iteration_i/max_iterations) + t

# MIAVOA Improved (Eq 25):
F = (2×rand1 + 1) × z × (-exp((iteration_i/max_iterations × (e-1))² - 1)) + t
```
The improved formula makes F decrease non-linearly, better balancing exploration/exploitation.

### 3.3 Exploration Phase (|F| ≥ 1)
```
# Strategy 1 (Eq 5-6):
D(i) = |X × R(i) - P(i)|       # X = 2×rand
P(i+1) = R(i) - D(i) × F

# Strategy 2 (Eq 7):
P(i+1) = R(i) - F + rand2 × ((ub - lb) × rand3 + lb)

# Selection (Eq 8): use Strategy 1 if P1 ≥ randP1, else Strategy 2
```

### 3.4 Exploitation Phase (|F| < 1)

#### Sub-phase 1 (0.5 ≤ |F| < 1):
```
# Food protection (Eq 10-11):
d(t) = R(i) - P(i)
P(i+1) = D(i) × (F + rand4) - d(t)

# Spiral flight (Eq 12-13) + Elite Candidate Pool (Eq 24):
S1 = R(i) × (rand5/(2π)) × cos(P(i))
S2 = R(i) × (rand5/(2π)) × sin(P(i))
P(i+1) = R(i) - (S1 + S2)
# MIAVOA adds: generate R1=R(i)+S1, R2=R(i)-S1, R3=R(i)+S2, R4=R(i)-S2
# Pick best among {R1,R2,R3,R4}

# Selection (Eq 9): use food protection if P2 ≥ randP2, else spiral
```

#### Sub-phase 2 (|F| < 0.5):
```
# Accumulation (Eq 15-16) + Adaptive Control (Eq 22-23):
A1 = BestVulture1(i) × (BestVulture1(i)×P(i)) / (BestVulture1(i) - P(i)²) × F
A2 = BestVulture2(i) × (BestVulture2(i)×P(i)) / (BestVulture2(i) - P(i)²) × F
# MIAVOA adaptive control:
α1 = α + (1-α) × (iter/max_iter)
α2 = (1-α) - (1-α) × (iter/max_iter)
P(i+1) = α1×A1 + α2×A2

# Competition / Lévy flight (Eq 17-18):
P(i+1) = R(i) - |d(t)| × F × Levy(Dim)
LF(x) = 0.01 × u×σ / |v|^(1/β)

# Selection (Eq 14): use accumulation if P3 ≥ randP3, else Lévy
```

### 3.5 Gaussian Quasi-Reflection Initialization (Eq 19-21)
```
# Gaussian chaos map (Eq 19):
x_{n+1} = 0 if x_n = 0, else (1/x_n) mod 1

# GQRBL (Eq 21):
x_gqo = rand((lb+ub)/2, lb+ub - γ×x)
# where γ is from Gaussian chaos mapping
# Keep whichever of x or x_gqo has better fitness
```

### Adaptation for 2D Waypoint Paths

All equations operate on `X = [x1,y1,x2,y2,...,xn,yn]` (vector of length 2n=16). The position vector IS the path. No decoding needed.

**Boundary clamping**: After every position update, clamp each coordinate:
```python
X = np.clip(X, [0,0,...], [WORLD_WIDTH, WORLD_HEIGHT,...])
```

---

## 4. Objective Functions — Concrete Formulas

### Objective 1: Time (`time_obj.py`)
```
T(path) = (1/v) × Σ ||P_{i+1} - P_i||
```
Pure Euclidean path length ÷ cruise speed.

### Objective 2: Energy (`energy_obj.py`)
```
E(path) = k1 × D_total + k2 × Σ_z(I_z × d_z(path))
```
Requires `wind.py` → `wind_exposure(path)` to compute `d_z` per zone.

### Objective 3: Risk (`risk_obj.py`)
```
R_clear = Σ_segments exp(-α × c_min(segment))
R_turn  = Σ_waypoints arccos(dot(v_in, v_out) / (|v_in|×|v_out|))
R_wind  = Σ_z(I_z × d_z(path))

R(path) = w_c × R_clear + w_t × R_turn + w_w × R_wind
```
Requires `collision.py` → `minimum_clearance(segment)` and `wind.py` → `wind_exposure(path)`.

---

## 5. Key Module Specifications

### `environment/collision.py`
- `segments_intersect(p1, p2, p3, p4) → bool` — basic segment intersection
- `segment_intersects_rect(seg_start, seg_end, rect) → bool` — segment vs AABB
- `point_in_rect(point, rect) → bool`
- `minimum_clearance_to_rect(seg_start, seg_end, rect) → float` — min distance from segment to rectangle
- `path_is_feasible(path, buildings) → bool` — check all segments vs all buildings
- `segment_is_feasible(p1, p2, buildings) → bool` — single segment check

### `environment/wind.py`
- `clip_segment_to_rect(seg_start, seg_end, rect) → float` — length of segment inside rectangle (Liang-Barsky or Cohen-Sutherland clipping)
- `wind_exposure(path, wind_zones) → dict[str, float]` — total distance inside each zone intensity level

### `optimizer/archive.py`
- `dominates(a, b) → bool` — Pareto dominance (all ≤, at least one <)
- `compute_crowding_distances(archive) → list[float]`
- `try_insert(candidate, archive) → bool` — full insert pipeline from design doc Section 5.2
- `prune_archive(archive, max_size)` — remove smallest crowding distance member

### `optimizer/crossover.py`
- `splice_crossover(parent1, parent2, env) → Path|None`
- `repair_joint(waypoint_a, waypoint_b, obstacle, env) → (new_a, new_b)|None`
- `tangent_snap(wp, obstacle, clearance) → new_wp|None`
- `iterative_pushout(wp_a, wp_b, obstacle, max_iters) → (new_a, new_b)|None`

---

## 6. Implementation Phases

### Phase A: Environment Foundation
1. `config.py` — all defaults
2. `environment/models.py` — dataclasses
3. `environment/generator.py` — random city generation
4. `environment/collision.py` — geometry utilities
5. `environment/wind.py` — zone clipping

### Phase B: Objectives
6. `objectives/time_obj.py`
7. `objectives/energy_obj.py`
8. `objectives/risk_obj.py`
9. `objectives/evaluator.py`

### Phase C: Optimizer Core
10. `optimizer/archive.py` — Pareto archive
11. `optimizer/leader_selection.py` — tournament
12. `optimizer/miavoa_core.py` — base equations

### Phase D: GA Component
13. `optimizer/crossover.py` — splice + repair
14. `optimizer/mutation.py` — jitter + detour
15. `optimizer/mo_miavoa.py` — main loop

### Phase E: Output
16. `selection/final_selection.py`
17. `visualization/plotting.py`
18. `main.py`

### Phase F: Testing
19. Unit tests for geometry, objectives, archive
20. Integration test: full run on a seeded environment

---

## 7. Verification Plan

### Automated Tests
- **Geometry**: known segment-rect intersections, clearance distances
- **Objectives**: hand-computed T, E, R for a simple 2-waypoint path
- **Archive**: verify dominance, crowding distance, insert/prune behavior
- **Full run**: seed=42, verify convergence (archive size grows, objectives improve)

### Visual Verification
- Plot environment with buildings + wind zones
- Overlay Pareto-front paths (color-coded by objective trade-off)
- Plot 3D Pareto front (T vs E vs R scatter)

---

## Open Questions

> [!IMPORTANT]
> **Language**: I'm planning Python with NumPy. Any preference for a different language or additional libraries?

> [!IMPORTANT]  
> **Visualization**: Matplotlib is the natural choice. Do you want interactive plots (Plotly) instead, or is static Matplotlib fine?

> [!NOTE]
> **The base paper's tables/figures were garbled in the docx conversion** (reversed text, broken formatting). I extracted all the equations and algorithm logic successfully. The experimental result tables weren't needed — only the method sections matter for implementation.
