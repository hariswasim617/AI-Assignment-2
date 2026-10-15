"""
All settings of the project in one place.

Every number used by the simulation, the fitness function and the Genetic
Algorithm (GA) is defined here as a plain Python variable. To change an
experiment you only edit this file (or override a value from the command line
in ``main.py``).

Units used everywhere:
    * distance on the grid  -> cells (1 cell = CELL_SIZE_M metres)
    * real distance         -> metres (m)
    * speed                 -> metres per second (m/s)
    * time                  -> seconds (s)
    * energy                -> joules (J)
"""

# ---------------------------------------------------------------------------
# 1. The world (grid) the drones fly over
# ---------------------------------------------------------------------------
GRID_SIZE = 100            # the map is GRID_SIZE x GRID_SIZE cells
CELL_SIZE_M = 10.0         # each cell is 10 m x 10 m, so the map is 1 km x 1 km
ENV_SEED = 7               # fixed seed -> every experiment uses the SAME map

N_NOFLY_ZONES = 7          # restricted airspace blocks (airport, buildings, ...)
N_WIND_POCKETS = 4         # Gaussian "high wind" hazard pockets
N_RADIATION_POCKETS = 3    # Gaussian "radiation" hazard pockets
N_PRIORITY_ZONES = 5       # disaster zones that we most want to survey
BACKGROUND_PRIORITY = 0.05 # every normal cell still has a small value
N_TARGETS = 30             # people / objects to detect on the ground

# Launch sites (x, y) in cell coordinates. The chromosome chooses one per drone.
BASE_STATIONS = [(6.5, 6.5), (93.5, 15.5), (48.5, 94.5)]

# ---------------------------------------------------------------------------
# 2. The drones (UAVs)
# ---------------------------------------------------------------------------
N_DRONES = 4
MAX_RANGE_M = 3000.0       # hard limit: a drone may fly at most 3 km
BATTERY_J = 60_000.0       # hard limit: 60 kJ (about 17 Wh) per drone
V_MIN = 3.0                # slowest allowed speed (m/s)
V_MAX = 18.0               # fastest allowed speed (m/s)
V_RETURN = 14.0            # speed used on the final "return to base" leg

SENSOR_RADIUS_MIN = 1.5    # sensor footprint radius (cells) at power 0
SENSOR_RADIUS_MAX = 5.0    # sensor footprint radius (cells) at power 1
SENSOR_POWER_MIN_W = 10.0  # electrical power of the sensor at power 0
SENSOR_POWER_MAX_W = 60.0  # electrical power of the sensor at power 1
WIND_POWER_FACTOR = 0.6    # flying in full wind costs 60 % more power
DETECTION_SPEED_FACTOR = 0.5  # at V_MAX the camera is 50 % less effective (motion blur)

# Rotary-wing power model of Zeng, Xu & Zhang (2019). These constants describe
# a typical small quadcopter. See fitness.flight_power() for the formula.
BLADE_PROFILE_POWER = 79.86   # P0 (W)
INDUCED_POWER = 88.63         # Pi (W)
ROTOR_TIP_SPEED = 120.0       # U_tip (m/s)
MEAN_ROTOR_VELOCITY = 4.03    # v0 (m/s)
FUSELAGE_DRAG_RATIO = 0.6     # d0
AIR_DENSITY = 1.225           # rho (kg/m^3)
ROTOR_SOLIDITY = 0.05         # s
ROTOR_DISC_AREA = 0.503       # A (m^2)

# ---------------------------------------------------------------------------
# 3. Chromosome shape
# ---------------------------------------------------------------------------
K_INIT_MIN = 6             # a new random drone gets 6..10 waypoints
K_INIT_MAX = 10
K_MIN = 2                  # mutation never deletes below 2 waypoints
K_MAX = 14                 # mutation never inserts above 14 waypoints
INIT_SENSOR_ON_PROB = 0.7  # chance that a random sensor bit starts as "on"

# ---------------------------------------------------------------------------
# 4. Fitness function
#    F = w1*Coverage + w2*Detection - w3*Energy - w4*Time - w5*Hazard - Penalty
# ---------------------------------------------------------------------------
W_COVERAGE = 0.35
W_DETECTION = 0.25
W_ENERGY = 0.15
W_TIME = 0.10
W_HAZARD = 0.15

# Constraint handling: "repair", "penalty" or "both"
CONSTRAINT_MODE = "both"
PENALTY_START = 2.0        # penalty multiplier at generation 0 ...
PENALTY_GROWTH = 4.0       # ... grows linearly to 2 * (1 + 4) = 10 at the last generation
NOFLY_WEIGHT = 10.0        # a no-fly incursion counts 10x more than a range overrun

# ---------------------------------------------------------------------------
# 5. Genetic Algorithm
# ---------------------------------------------------------------------------
SEED = 42
POP_SIZE = 100
GENERATIONS = 200
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.1        # probability that one gene is mutated
SELECTION = "tournament"   # "tournament", "rank" or "roulette"
TOURNAMENT_K = 3           # selection pressure of tournament selection
RANK_PRESSURE = 1.7        # selection pressure of linear rank selection (1..2)
ELITISM = 0.05             # top 5 % are copied unchanged to the next generation

BLX_ALPHA = 0.3            # how far BLX-alpha may go outside the parents
SEGMENT_SWAP_PROB = 0.5    # chance to take each drone segment from the other parent
SIGMA_START = 8.0          # Gaussian waypoint mutation step (cells) at the start ...
SIGMA_END = 1.0            # ... shrinks to this at the end (annealing)
RESET_PROB = 0.02          # small chance to re-create a whole drone at random

DIVERSITY_THRESHOLD = 0.3  # if diversity < 30 % of the starting diversity ...
IMMIGRANT_FRACTION = 0.1   # ... replace 10 % of the population with random newcomers
MUTATION_BOOST = 2.0       # ... and double the mutation rate for that generation

STAGNATION_PATIENCE = 60   # stop if the best fitness did not improve ...
STAGNATION_EPS = 1e-4      # ... by more than this for 60 generations
TARGET_FITNESS = None      # stop early if this fitness is reached (None = off)

# ---------------------------------------------------------------------------
# 6. Experiments
# ---------------------------------------------------------------------------
RESULTS_DIR = "results"
SWEEP_SEEDS = [1, 2, 3]            # seeds for each sensitivity setting
SWEEP_GENERATIONS = 100            # generations per sensitivity run
SWEEP_POP_SIZES = [20, 50, 100, 200]
SWEEP_MUTATION_RATES = [0.01, 0.05, 0.1, 0.2, 0.4]
SWEEP_TOURNAMENT_K = [2, 3, 5, 8]
SWEEP_SELECTIONS = ["roulette", "rank", "tournament"]
SWEEP_CONSTRAINT_MODES = ["penalty", "repair", "both"]
BASELINE_SEEDS = [1, 2, 3, 4, 5]   # seeds for GA vs. random search


def as_dict() -> dict:
    """Return every UPPER_CASE setting of this module as a dictionary.

    This is used to save the exact settings of a run next to its results,
    so every figure can be traced back to the numbers that produced it.
    """
    return {name: value for name, value in globals().items()
            if name.isupper() and not name.startswith("_")}
