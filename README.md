# Multi-Objective Green Urban Logistics: Drone Fleet Path Planning

An advanced, high-performance optimization framework extending the Multi-Strategy Improved African Vulture Optimization Algorithm (MIAVOA) into a Multi-Objective engine (MO-MIAVOA) augmented with a Genetic Algorithm (GA) fusion layer. 

This project solves the real-world problem of autonomous drone fleet routing in smart cities by simultaneously optimizing delivery speed, battery depletion under variable wind fields, and structural airspace risk.

---

## 1. Project Overview & Real-World Use Case

Standard navigation models optimize for a single objective: distance or time. In real-world urban logistics, an autonomous commercial drone fleet faces severely conflicting constraints. Shorter paths might fight heavy wind resistance, dramatically draining batteries, while the safest air corridors might wind around city hazards, significantly increasing customer wait times. 

This project models a 3D urban workspace populated with static hazards (no-fly zones, buildings) and dynamic environmental obstacles (crosswind vector fields). The goal is to compute a **Pareto Optimal Front**—a family of optimal compromise paths that allows fleet operators to balance speed, energy efficiency, and safety.

---

## 2. Core Optimization Keywords Explained

To surpass the limitations of the baseline paper, this project introduces four advanced architectural upgrades to improve search robustness and escape localized traps:

### I. Multi-Objective Optimization (MOO)
* **What it is:** A mathematical framework that handles a vector of multiple conflicting objective functions instead of compressing them into a single score.
* **Performance Improvement:** Rather than locking onto a single path, the algorithm populates and updates a fixed-size **Pareto Archive**. This preserves structural diversity within the vulture flock, ensuring the algorithm maintains multiple optimal navigation routes concurrently.

### II. Adaptive Lévy Flight
* **What it is:** An exploration mechanism where step sizes scale dynamically based on the "crowding distance" of the Pareto Front instead of remaining rigid.
* **Performance Improvement:** When multiple drone path vectors begin crowding into the exact same local coordinate corridor (a mathematical bottleneck), the algorithm senses this high-density crowding and scales up the Lévy step length ($\sigma$). This aggressively pushes trapped drones out of localized dead-ends, forcing them to discover unmapped, clear airways.

### III. Chaotic Maps
* **What it is:** Deterministic, pseudo-random mathematical sequences used to scatter agents through complex search landscapes.
* **Performance Improvement:** We implement a runtime chaotic map injection layer. If the urban environment encounters sudden dynamic disruptions—such as a building crane moving or a micro-weather system geofencing a sector—the chaotic map instantly re-initializes a subset of the population, scattering them cleanly to navigate around the newly emerged obstacle without stalling.

### IV. Genetic Algorithm (GA) Fusion
* **What it is:** An evolutionary layer that applies selection, crossover, and mutation operators to high-performing coordinate vectors.
* **Performance Improvement:** In the base paper, late-stage exploitation relies on testing small, localized coordinate offsets. We replace this with a GA crossover operator. The algorithm treats elite paths from the Pareto Archive as "chromosomes." By cross-breeding a route containing high speed with a route containing exceptional wind shielding, the algorithm breeds entirely new hybrid "super-routes" that standard vulture flight equations could never organically stumble upon.

---

## 3. Mathematical Multi-Objective Formulations

The algorithm acts as a minimization engine operating on three conflicting objective scores ($f_1, f_2, f_3$) for any candidate path $\mathbf{P}$:

### Objective 1: Delivery Time Minimization ($f_1$)
Optimizes path velocity profiles to ensure prompt package arrival.
$$f_1(\mathbf{P}) = \sum_{k=1}^{N-1} \frac{\|\mathbf{p}_{k+1} - \mathbf{p}_k\|}{V_{\text{drone}}}$$

### Objective 2: Battery & Energy Conservation ($f_2$)
Models aerodynamic power draw by factoring in drone velocity relative to a dynamic wind field vector ($\vec{W}$). Flying directly against a headwind exponentially penalizes this function.
$$f_2(\mathbf{P}) = \sum_{k=1}^{N-1} \text{Power}\left(\vec{V}_{\text{drone}}, \vec{W}(\mathbf{p}_k)\right) \times \Delta t_k$$

### Objective 3: Airspace Risk Minimization ($f_3$)
Imposes an exponential proximity penalty on any path coordinate that approaches geofenced zones (airports, hospitals, dense residential grids).
$$f_3(\mathbf{P}) = \sum_{k=1}^{N} \sum_{m=1}^{M} \frac{\gamma}{\|\mathbf{p}_k - \mathbf{Hazard}_m\|^2}$$

---

## 4. Architectural Assumptions & Modifications

### System Assumptions
1. **Search Bound Constraints:** The 3D airspace grid is strictly bounded to a standardized coordinate limit of $X, Y \in [-100, 100]$ and $Z \in [0, 50]$. Any node violating these parameters is strictly clipped back to boundary conditions via a boundary enforcement function.
2. **Static Environment:** Hazards and no-fly structures are modeled as deterministic 3D geometric primitives (spheres and cylinders).
3. **Continuous Path Representation:** A drone flight path is modeled as a continuous array of 3D waypoint vertices.

### Algorithmic Improvements Over Base Paper
* **Leader Selection:** Baseline MIAVOA selects two absolute scalar leaders (`BestVulture1` and `BestVulture2`). This project replaces them with a **Leader Committee** sampled directly from the non-dominated Pareto Front using tournament selection based on crowding distance.
* **Exploitation Overhaul:** Shifting focus from basic localized clustering equations to the GA crossover framework during the late-stage optimization phase ($|F| < 0.5$).

---

## 5. Phased Implementation Roadmap
### Phase 1: Standalone Strategy Validation (C++)
* [x] Implement **Strategy 2: Adaptive Control Strategy** with time-varying weight functions ($\alpha_1, \alpha_2$).
* [ ] Verify module independently against standard mathematical test functions to validate mathematical convergence stability.
* [ ] Isolate and test individual teammate sub-modules (GQRL, ECPS, IHP).

### Phase 2: Multi-Objective Core Architecture (Python)
* [ ] Bind compiled, high-performance C++ vulture movement modules into an executive Python core framework.
* [ ] Construct the 3D smart city simulation environment, mapping the wind fields and geofenced zones.
* [ ] Implement the Multi-Objective engine: Pareto dominance check logic, sorting routines, and external archiving controls.
* [ ] Wire in the Adaptive Lévy Flight scaling and the GA Crossover/Mutation route-breeding loops.

### Phase 3: Benchmarking & Analytical Performance Evaluation
* [ ] Execute the full `MO-MIAVOA` code across standard validation test suites (ZDT1-ZDT6, DTLZ1-DTLZ4).
* [ ] Benchmark performance metrics directly against industry gold-standards: **NSGA-II, MOEA/D, and MOPSO**.
* [ ] Evaluate quality indicators using Inverted Generational Distance (IGD), Hypervolume (HV), and Spread graphs.
* [ ] Generate publication-grade 3D plots showing the optimized autonomous drone pathways navigating safely across the city landscape.

---

## 6. Project Setup & Local Execution

### Prerequisites
* GCC Compiler supporting **C++17** or higher
* Python **3.10+**
* OpenMP (Optional, for multi-threaded CPU acceleration)

### Building the Phase 1 Standalone Module
```bash
# Compile the validated local execution script
g++ -O3 avoa2_exploitation.cpp -o avoa2_solver.exe

# Execute the local binary
.\avoa2_solver.exe