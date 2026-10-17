# Learning Guide (simple explanation)

## The idea in 5 lines
1. The map is a 50 x 50 grid with no-fly zones (black), hazard zones (orange), priority zones (blue boxes) and targets (x).
2. 3 drones start at the base, each visits 6 waypoints and flies back.
3. A **chromosome** is one full plan: for every waypoint `[x, y, speed, sensor_power]`.
4. The **fitness** gives a score: good = high coverage and detection, bad = energy, time, hazard and breaking rules.
5. The **GA** starts with 50 random plans and improves them for 100 generations using selection, crossover and mutation.

## The files
* `config.py` – all numbers (change them here).
* `environment.py` – builds the map. `nearest_free()` finds the closest open cell (used by repair).
* `chromosome.py` – the `Chromosome` class. `random()` makes a random plan, `route()` gives base → waypoints → base.
* `fitness.py` – `repair()` fixes waypoints inside no-fly zones; `evaluate()` computes the 5 objectives, the penalty and the fitness.
* `ga.py` – the `GeneticAlgorithm` class:
  `initial_population()`, `tournament_selection()`, `crossover()`, `mutate()`, `run()` (with elitism, immigrants and the stop rules).
* `random_search.py` – makes random plans and keeps the best (the baseline).
* `plots.py` – draws all figures. `main.py` – runs the 3 experiments.

## 15 likely viva questions
1. **Why a GA?** Many conflicting goals and a huge search space; a GA only needs a fitness score and searches many plans at once.
2. **What is your chromosome?** A real-valued array (3 drones x 6 waypoints x 4 genes): x, y, speed, sensor power.
3. **Why real numbers?** Positions are continuous; a small gene change = a small route change, good for mutation and blend crossover.
4. **Write your fitness function.** F = 0.35·Coverage + 0.25·Detection − 0.15·Energy − 0.10·Time − 0.15·Hazard − 2·Penalty.
5. **Why these weights?** Mission goals get 0.60, costs 0.40; coverage is the main goal; safety and energy equal; time smallest because range already limits it.
6. **Why scale objectives to 0–1?** So no objective dominates just because its numbers are bigger.
7. **How do you handle constraints?** Repair (move waypoints out of no-fly zones) and penalty (path through a no-fly zone, or too long).
8. **How does tournament selection work?** Pick k random plans, the best becomes a parent. Bigger k = more selection pressure.
9. **Explain your crossover.** Segment swap (whole drone routes from either parent) + geometric blend (one drone's waypoints mixed between both parents).
10. **Explain your mutation.** Each waypoint (10% chance) moves by a small random Gaussian step, speed and power change a little; sometimes two waypoints swap order.
11. **What is premature convergence?** The population becomes the same too early and gets stuck. Mutation, immigrants and moderate k prevent it (mutation 0.02 and k = 5 got stuck in my tests).
12. **What is elitism?** The 2 best plans are copied unchanged, so the best fitness never gets worse.
13. **When does the GA stop?** After 100 generations, or after 30 generations without improvement.
14. **What did the sensitivity analysis show?** Bigger population = better but slower; mutation 0.02 too low; k = 2 best, k = 5 worst (too much pressure).
15. **Why is the GA better than random search?** Same number of evaluations, but the GA keeps and combines good routes; random search never found a plan without breaking rules.
