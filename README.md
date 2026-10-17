# UAV Fleet Planning with a Genetic Algorithm

**Artificial Intelligence – BSCS 5 · Assignment 2 · Bahria University Lahore Campus**
**Author:** [Your Name] – [Roll No]

A Genetic Algorithm (written from scratch in Python) that finds the **deployment positions, waypoint
routes, speeds and sensor activation schedules** of **3 drones** surveying a **50 x 50 hazard grid**
with no-fly zones, hazard zones (wind / radiation), disaster (priority) zones and targets.

![Flight paths across generations](results/paths_generations.png)

## Run it in Google Colab (easiest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hariswasim617/AI-Assignment-2/blob/main/UAV_GA_Colab.ipynb)

1. Click the badge above, **or** go to [colab.research.google.com](https://colab.research.google.com) →
   *File → Upload notebook* → choose `UAV_GA_Colab.ipynb` from this repository.
2. Click **Runtime → Run all**. Nothing needs to be installed.
3. The figures appear under each cell (about 8 minutes in total) and are saved in the `results` folder.

The notebook contains the same code as the `.py` files, one file per cell.

## Files
| File | What it does |
|---|---|
| `config.py` | all settings (map, drones, fitness weights, GA settings) |
| `environment.py` | the grid: no-fly zones, hazard zones, priority zones, targets |
| `chromosome.py` | `Chromosome` class: deployment position + waypoints, speed, sensor power and sensor on/off for every drone |
| `fitness.py` | multi-objective fitness function + constraint handling (repair and penalty) |
| `ga.py` | `GeneticAlgorithm` class: population initialisation, tournament selection, crossover, mutation, termination |
| `random_search.py` | baseline random search for the comparison |
| `plots.py` | all figures |
| `main.py` | runs the simulation and all experiments, saves the results in `results/` |
| `results/` | figures (PNG) and result tables (CSV) |
| `report/` | technical report (Word + PDF) |
| `UAV_GA_Colab.ipynb` | the whole project as a Google Colab notebook (same code, one file per cell) |

## Run it on your own computer
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python main.py                     # about 10 minutes; all figures and tables go to results/
```
To try other settings, change the numbers in `config.py` and run `python main.py` again.

## Design
**Chromosome** (hybrid real + binary, one segment per drone):
```
[ deploy_x, deploy_y | (x, y, speed, power, on) x 6 waypoints ]
```
`deploy_x, deploy_y` = where the drone is launched and lands, `x, y` = waypoint, `speed` = speed on the
leg to that waypoint, `power` = sensor power (radius 1–4 cells), `on` = sensor on/off on that leg.
Each drone flies deployment position → 6 waypoints → deployment position.

**Fitness** (every part scaled to 0..1):
```
F = 0.35*Coverage + 0.25*Detection - 0.15*Energy - 0.10*Time - 0.15*Hazard - 2*Penalty
```
* maximise: priority-weighted coverage of the disaster zones, target detection rate
* minimise: energy, flight time (slowest drone), hazard exposure
* penalty: flying through a no-fly zone or further than the maximum range (150 cells)

**Constraint handling** – repair (deployment positions / waypoints inside a no-fly zone are moved to the
nearest open cell) + penalty function.

**Genetic operators** – tournament selection (k = 3) with elitism (2); crossover = drone-segment swap +
geometric blend; mutation = Gaussian move of waypoints / deployment positions, small speed and power
changes, sensor bit flip and waypoint swap; 2 random immigrants per generation keep the population
diverse. **Termination:** 100 generations or 30 generations without improvement.

## Results
**Main run** (population 50, 100 generations, seed 42): best fitness **-2.409 → 0.391**.
The final plan covers 0.75 of the priority-weighted area, finds 100% of the targets and breaks no rules.

| Fitness convergence (best vs. average) | Coverage heatmaps across generations |
|---|---|
| ![](results/convergence.png) | ![](results/coverage_heatmaps.png) |

**Hyperparameter sensitivity** (final best fitness, mean ± std over 3 seeds):

| Setting | Value 1 | Value 2 | Value 3 |
|---|---|---|---|
| Population size | 20: 0.338 ± 0.044 | 50: 0.396 ± 0.008 | 100: 0.415 ± 0.004 |
| Mutation rate | 0.02: 0.393 ± 0.002 | 0.1: 0.396 ± 0.008 | 0.3: 0.366 ± 0.012 |
| Selection pressure (tournament k) | 2: 0.366 ± 0.039 | 3: 0.396 ± 0.008 | 5: 0.407 ± 0.004 |

![Sensitivity](results/sensitivity.png)

**GA vs. random search** (same number of fitness evaluations, 5 seeds):

| | GA | Random search |
|---|---|---|
| Fitness | **0.374 ± 0.039** | -0.859 ± 0.126 |
| Coverage | 0.707 ± 0.063 | 0.279 ± 0.140 |
| Detection | 0.947 ± 0.107 | 0.320 ± 0.176 |
| Energy | 0.367 ± 0.041 | 0.484 ± 0.032 |
| Time | 0.375 ± 0.052 | 0.560 ± 0.112 |
| Hazard | 0.115 ± 0.014 | 0.081 ± 0.033 |
| Penalty | 0.000 ± 0.000 | 0.448 ± 0.058 |

![GA vs random](results/ga_vs_random.png)
![GA vs random maps](results/ga_vs_random_maps.png)

The full discussion is in [`report/Technical_Report.pdf`](report/Technical_Report.pdf).
