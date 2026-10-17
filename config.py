"""
All settings of the project (plain variables).
Change a number here and run `python main.py` again.
"""

# ---------------- The map (a 50 x 50 grid, 1 cell = 20 m) ----------------
GRID_SIZE = 50
BASE = (2.0, 2.0)          # all drones take off from here and come back here
MAP_SEED = 7               # fixed seed -> the targets are always in the same place

# No-fly zones as rectangles (x1, y1, x2, y2): cells x1..x2-1, y1..y2-1 are blocked
NO_FLY_ZONES = [(12, 8, 18, 30), (28, 0, 34, 14), (24, 30, 36, 36)]

# Hazard zones (high wind / radiation) as circles (centre_x, centre_y, radius)
HAZARD_ZONES = [(42, 24, 8), (22, 20, 7), (10, 40, 6)]

# Disaster (priority) zones as rectangles (x1, y1, x2, y2, importance 0..1)
PRIORITY_ZONES = [(36, 18, 48, 30, 1.0), (4, 34, 16, 46, 0.8), (20, 40, 30, 48, 0.6)]
BACKGROUND_PRIORITY = 0.1  # every other cell is a little bit useful too
N_TARGETS = 15             # targets placed inside the priority zones

# ---------------- The drones ----------------
N_DRONES = 3
N_WAYPOINTS = 6            # waypoints per drone
SPEED_MIN, SPEED_MAX = 1.0, 3.0   # cells per time step
MAX_RANGE = 200            # max path length per drone (cells), including the way back

# ---------------- Fitness weights ----------------
W_COVERAGE = 0.35
W_DETECTION = 0.25
W_ENERGY = 0.15
W_TIME = 0.10
W_HAZARD = 0.15
PENALTY_WEIGHT = 2.0       # how hard constraint violations are punished

# ---------------- Genetic Algorithm ----------------
SEED = 42
POP_SIZE = 50
GENERATIONS = 100
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1        # chance that one waypoint is mutated
MUTATION_STEP = 3.0        # how far (cells) a mutated waypoint moves on average
TOURNAMENT_K = 3           # tournament size = selection pressure
ELITE = 2                  # best individuals copied unchanged to the next generation
IMMIGRANTS = 2             # new random individuals per generation (keeps diversity)
PATIENCE = 30              # stop if the best fitness did not improve for 30 generations
