# Learning Guide – how this project works (in plain English)

This guide explains the whole project so you can understand it, change it and explain it in a viva.
Read it next to the code: every file below also has comments next to the important lines.

---

## 1. The problem in one paragraph

A company has **4 drones** and must survey a **1 km × 1 km disaster area**. The area is a
**100 × 100 grid** (one cell = 10 m). Some cells are **no-fly zones** (drones may never enter them),
some cells are **dangerous** (strong wind or radiation), and some cells are **important**
(disaster zones where people may be trapped). There are also **30 targets** to find.
We must decide, for every drone: where it takes off, which points (waypoints) it flies to,
how fast it flies on each leg, and when its sensor is switched on and how strongly.
There is no single "correct" answer – a plan that covers more area uses more battery, a plan that
avoids danger may miss important zones. So we use a **Genetic Algorithm (GA)** to search for a good
compromise.

## 2. What a Genetic Algorithm does (the big picture)

A GA copies natural evolution:

1. **Population** – start with 100 random plans ("individuals").
2. **Fitness** – give every plan a score (higher = better).
3. **Selection** – better plans are more likely to become parents.
4. **Crossover** – two parents are mixed to create children.
5. **Mutation** – children get small random changes (so new ideas keep appearing).
6. **Repair** – children that break the rules (e.g. fly through a no-fly zone) are fixed.
7. **Elitism** – the best 5 % are copied unchanged, so the best plan is never lost.
8. Repeat steps 2–7 for 200 **generations**, or stop earlier if nothing improves any more.

## 3. The files, one by one

| File | What it does |
|---|---|
| `uav_ga/config.py` | **All settings** as plain variables (grid size, drone limits, fitness weights, GA settings, experiment settings). Change numbers here. |
| `uav_ga/environment.py` | Builds the **map**: no-fly zones, hazard field, priority map, targets, base stations. Also contains **BFS** (nearest free cell) and **A\*** (path around obstacles). |
| `uav_ga/chromosome.py` | The **chromosome** = one complete fleet plan. One *segment* per drone. |
| `uav_ga/fitness.py` | **Simulates** the flight (distance, time, energy, hazard) and computes the **fitness**. Also contains the drone **energy model**. |
| `uav_ga/constraints.py` | **Repair**: moves waypoints out of no-fly zones, adds detours, shortens routes that are too long. |
| `uav_ga/operators.py` | **Selection** (tournament, rank, roulette), **crossover** (segment swap + BLX-α + uniform bits), **mutation** (6 kinds). |
| `uav_ga/ga.py` | The **GA loop**: initialise, evaluate, select, crossover, mutate, repair, elitism, diversity control, termination, history. |
| `uav_ga/baseline.py` | **Random search** – the comparison method. |
| `uav_ga/visualization.py` | All **plots** (maps, convergence curves, heatmaps). |
| `main.py` | Runs the **experiments** from the command line and saves the results. |
| `tests/test_uav_ga.py` | **Unit tests** (`pytest`). |
| `UAV_GA_Colab.ipynb` | The **same code in notebook cells** for Google Colab. |
| `report/build_report.py` | Builds the Word/PDF report from the saved results. |

### 3.1 `config.py` – settings
Only variables, no logic. Example: `POP_SIZE = 100`, `MUTATION_RATE = 0.01`, `W_COVERAGE = 0.35`.
Other files read them as `cfg.POP_SIZE`, so every number is defined in exactly one place.

### 3.2 `environment.py` – the map
* `blocked[y, x]` – True for no-fly cells. 7 random rectangles/circles, kept away from the base stations.
* `hazard[y, x]` – 0..1. Made of Gaussian "bumps" (`gaussian_blob`): 4 wind pockets and 3 radiation
  pockets. Two radiation pockets are put **near disaster zones on purpose** so the GA faces a real
  trade-off (the important place is also the dangerous place).
* `priority[y, x]` – 0..1. Five disaster zones (Gaussian bumps with weight 0.5–1); everywhere else 0.05.
* `targets` – 30 points, 75 % inside disaster zones.
* `blocked_margin` – the no-fly mask grown by 1 cell, so repaired routes keep a safety distance.
* `_nearest_free_cells()` – a **multi-source BFS**: starting from all free cells at once, it finds
  for every blocked cell its nearest free cell. Computed once → later it is a simple table lookup.
* `astar()` – classic **A\*** search with the straight-line distance as heuristic.
* `segment_is_clear(a, b)` – checks a straight line by testing points every 0.5 cells.
* `sample_legs()` – walks along all legs at once (NumPy) and reports their length, mean wind,
  mean hazard and how many samples were inside no-fly zones.

### 3.3 `chromosome.py` – the encoding
For each drone the chromosome stores:

```
[ base | (x1, y1, v1, p1, a1) | (x2, y2, v2, p2, a2) | ... | (xK, yK, vK, pK, aK) ]
```

* `base` – integer: which of the 3 base stations the drone starts from (and returns to).
* `x, y` – real: waypoint position.
* `v` – real: speed (3–18 m/s) on the leg that ends at this waypoint.
* `p` – real: sensor power 0..1 (bigger = larger footprint but more battery).
* `a` – binary: sensor on (1) or off (0) on that leg → together the **sensor schedule**.

The drone flies `base → w1 → … → wK → base`. K can differ per drone (6–10 at the start, 2–14 later).
This mix of integer, real and binary genes is why it is called a **hybrid multi-segment** encoding.
`path_features()` turns a plan into a fixed-length vector (each route resampled to 10 points) so we
can measure how different two plans are (used for **diversity**).

### 3.4 `fitness.py` – how good is a plan?
1. `build_legs()` – makes a table of all legs of all drones.
2. `simulate_flight()` – for every leg: length → time (= length / speed) → energy
   (= (flight power + sensor power) × time). Wind increases the flight power by up to 60 %.
   Hazard dose = mean hazard under the leg × time spent there.
3. `flight_power(v)` – the rotary-wing power model of Zeng et al. (2019): blade power + induced
   (lift) power + drag power. Flying very slowly is expensive (lots of lift power), flying fast costs
   drag; in our speed range faster is cheaper per metre.
4. `coverage_map()` – a cell is covered if its centre is within the sensor radius of a leg flown
   with the sensor **on**.
5. `detection_probabilities()` – each active leg has a chance to detect a target:
   `quality(power) × speed_factor(speed) × (1 − (d/r)²)`. Slow flight = sharper images.
   Several legs combine as `1 − Π(1 − P_leg)`.
6. `FitnessEvaluator.score()`:

```
F = 0.35·Coverage + 0.25·Detection − 0.15·Energy − 0.10·FlightTime − 0.15·Hazard − λ(g)·Violation
```

All five terms are between 0 and 1 (normalised), so the weights are fair.
`λ(g) = 2·(1 + 4·g/G)` grows from 2 to 10 – the **dynamic penalty**: mild early (explore),
strict at the end (must be feasible).

### 3.5 `constraints.py` – repair
`repair()` runs four steps on every new child:
1. `clip_genes` – put every gene back in its range.
2. `move_waypoints_out_of_nofly` – BFS table lookup.
3. `fix_crossing_legs` – for a leg that crosses a no-fly zone, `find_detour` tries one side point,
   then splits the problem in two (divide and conquer), and finally uses A\*.
4. `trim_to_limits` – if a drone cannot fly its route **and return home** within 3 km and 60 kJ,
   drop waypoints from the end of the route.

In `"penalty"` mode only step 1 runs and violations are punished; in `"both"` (default) repair runs
and the penalty catches anything left.

### 3.6 `operators.py` – selection, crossover, mutation
* **Tournament selection** – pick k random plans, the best one wins. k = selection pressure.
* **Rank selection** – chance depends on rank, not raw fitness.
* **Roulette selection** – chance proportional to (shifted) fitness.
* **Crossover** – (1) **segment swap**: a whole drone route is taken from parent 1 or parent 2
  (keeps good routes intact); (2) **BLX-α** blend on coordinates/speeds/powers (a geometric
  crossover – the child waypoint lies between the parents' waypoints); (3) **uniform crossover** on
  the sensor bits.
* **Mutation** – Gaussian move of waypoints (step size σ shrinks from 8 to 1 cell = *annealing*),
  speed jitter, power jitter, sensor bit flip, change base station, insert / delete / swap waypoints,
  and rarely (2 %) a completely new random route.

### 3.7 `ga.py` – the engine
`GeneticAlgorithm.run()` does the loop of section 2. Extra details:
* **Diversity control** – if the mean distance between plans drops below 30 % of the starting value,
  10 % random newcomers ("immigrants") are added and the mutation rate is doubled.
* **Termination** – max generations, OR no improvement > 0.0001 for 60 generations (stagnation),
  OR a target fitness. The reason is saved in `run_summary.json`.
* **History** – one row per generation (best, mean, std, diversity, objectives, …) → `history.csv`.

### 3.8 `baseline.py` – random search
Generates random plans with the **same** generator and **same** repair as the GA's first
generation, for **exactly the same number of fitness evaluations**, and keeps the best.
So if the GA wins, it is because of selection/crossover/mutation – not because it got more tries.

### 3.9 `main.py` – experiments
* `run_main` – one GA run (pop 100, 200 generations, seed 42) + all main figures.
* `run_sensitivity` – changes **one** parameter at a time (population size, mutation rate,
  tournament k, selection method, constraint mode), 3 seeds each, 100 generations.
* `run_baseline` – GA vs. random search, 5 seeds, mean ± std table.

## 4. How to run

```bash
pip install -r requirements.txt
python main.py --experiment all --quick   # 2-minute test of everything
python main.py                            # main run
python main.py --experiment sensitivity   # sweeps (about 1–1.5 hours)
python main.py --experiment baseline      # GA vs random search (about 40 minutes)
pytest -q                                 # tests
```
Or open `UAV_GA_Colab.ipynb` in Google Colab and run all cells.

## 5. The 15 most likely viva questions (with short answers)

1. **Why did you use a Genetic Algorithm instead of A\* or another exact method?**
   The problem has many conflicting objectives, mixed variable types (positions, speeds, on/off bits)
   and a huge search space (thousands of genes). There is no single goal state for A\*. A GA searches
   many good compromises at the same time and only needs a fitness function, not a gradient.

2. **Explain your chromosome.**
   One segment per drone: an integer base index, then K waypoints, each with x, y (real), speed
   (real), sensor power (real) and a sensor on/off bit (binary). It is hybrid (integer + real +
   binary) and variable-length (K changes by insert/delete mutation).

3. **Why real-valued coordinates and not a binary string?**
   Positions are continuous; with real numbers a small change in a gene means a small change in the
   route, so mutation and BLX-α crossover behave smoothly. A binary encoding would make one bit flip
   jump the drone across the map.

4. **Write the fitness function and justify the weights.**
   `F = 0.35·Cov + 0.25·Det − 0.15·Energy − 0.10·Time − 0.15·Hazard − λ(g)·V`. Coverage of
   disaster zones is the main mission goal (largest weight), detection is the next goal. Safety
   (hazard) and energy are equally important secondary costs; time gets the smallest weight because
   the range limit already bounds the mission length. The positive weights add up to 0.60 and the
   costs to 0.40, so the GA is pushed to do the mission, not just to stay at home.

5. **Why normalise every objective to 0..1?**
   Otherwise the term with the biggest unit (e.g. energy in joules) would dominate and the weights
   would be meaningless.

6. **What is the difference between repair and penalty?**
   Repair changes the chromosome so it becomes feasible (move waypoint, add detour, shorten route).
   Penalty keeps the infeasible chromosome but lowers its fitness. Repair wastes no evaluations on
   illegal plans; the penalty is simple and lets the GA pass through infeasible regions. I use both.

7. **What is a dynamic penalty and why use it?**
   The penalty factor grows with the generation (2 → 10). Early on, slightly illegal plans can survive
   and help exploration; at the end illegal plans are strongly punished, so the final answer is legal.

8. **How does BFS / A\* help here?**
   BFS (run once) gives the nearest free cell for every blocked cell → moving a waypoint out of a
   no-fly zone is a lookup. A\* finds a detour around no-fly zones when the simple side-point detour
   fails.

9. **Explain your crossover.**
   First, whole drone routes are swapped between the parents (segment swap – keeps good routes
   intact). Then, for some drones, real genes are blended with BLX-α (a child gene is drawn from the
   interval between the parents, widened by α = 0.3) and the sensor bits use uniform crossover.

10. **Explain your mutation and why sigma is annealed.**
    Six mutation types (move waypoint, speed, power, bit flip, base change, insert/delete/swap,
    random reset). The waypoint step σ shrinks from 8 to 1 cell: big steps early to explore, small
    steps late to fine-tune.

11. **What is premature convergence and how do you avoid it?**
    When the whole population becomes almost identical too early and gets stuck in a local optimum.
    I measure diversity (mean pairwise distance); if it falls below 30 % of the start value, I add
    random immigrants and double the mutation rate. Mutation, BLX-α's widening and moderate
    selection pressure also keep diversity.

12. **What is selection pressure? What did the sensitivity analysis show?**
    How strongly the best individuals are preferred. In tournament selection it is k. Small k =
    more exploration but slower progress; large k = fast progress but risk of premature convergence.
    Measured: k = 2 and 3 gave 0.295 / 0.294, k = 5 and 8 gave 0.326 / 0.325 (but k = 8 varied most
    between seeds and had the lowest diversity). The mutation rate mattered even more: 0.356 at
    p_m = 0.01 down to 0.262 at p_m = 0.4. So the final settings are p_m = 0.01 and k = 5.

13. **What does elitism do?**
    The best 5 % are copied unchanged into the next generation, so the best fitness can never go
    down.

14. **When does your GA stop?**
    After the maximum number of generations, or if the best fitness has not improved by more than
    0.0001 for 60 generations (stagnation), or if a target fitness is reached. The reason is logged.

15. **How did you make the comparison with random search fair?**
    Same map, same random plan generator, same repair, same fitness function (with the final
    penalty), and exactly the same number of fitness evaluations; 5 seeds; I report mean ± std.

## 6. Key results to quote

| What | Number |
|---|---|
| Main run best fitness | 0.150 (generation 0) → **0.381** (generation 200) |
| Final plan | coverage 0.83, 28 / 30 targets, 100 % of high-priority cells, 138 kJ, makespan 196 s, no violations |
| GA vs. random search (5 seeds, same 19,300 evaluations) | **0.383 ± 0.010 vs. 0.208 ± 0.009** – GA better on every metric |
| Best mutation rate / tournament size | p_m = 0.01 (0.356) / k = 5 (0.326); together 0.364 |
| Repair vs. penalty | repair 0.295 (100 % feasible) vs. penalty only 0.264 (54 % feasible), but penalty is ~4.5× faster |

## 7. Small things worth remembering
* Seeds: the map always uses seed 7; the main GA run uses seed 42; sweeps use seeds 1–3; the
  baseline uses seeds 1–5. Same seed → exactly the same result.
* "Lamarckian repair": the repaired genes are written back into the chromosome, so children inherit
  the fixed version.
* The return-to-base leg is not a gene: it always uses 14 m/s with the sensor off.
* A target counts as "detected" when its detection probability is at least 0.5.
