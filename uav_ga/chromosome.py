"""
The chromosome: one complete mission plan for the whole drone fleet.

Encoding (hybrid, multi-segment)
--------------------------------
The chromosome has one SEGMENT per drone. A segment mixes integer, real and
binary genes, which is why it is called a "hybrid" encoding:

    segment = [ base | (x1, y1, v1, p1, a1) | (x2, y2, v2, p2, a2) | ... | (xK, yK, vK, pK, aK) ]

    base    integer  index of the launch site (0 .. number of bases - 1)
    x, y    real     waypoint position on the grid (cells)
    v       real     speed on the leg that ENDS at this waypoint (m/s)
    p       real     sensor power on that leg (0..1); bigger power = bigger footprint
    a       binary   1 = sensor switched on during that leg, 0 = off
                     (all the a-bits together form the sensor activation schedule)

The drone flies   base -> w1 -> w2 -> ... -> wK -> base.
The last "return home" leg always uses speed V_RETURN with the sensor off.
K (the number of waypoints) can be different for every drone and can change
during evolution (insert/delete mutation), so the chromosome has a variable
length. Inside the code each gene type is stored in its own NumPy array,
which keeps the operators simple and fast.
"""

import numpy as np

from uav_ga import config as cfg
from uav_ga.environment import Environment


class DroneSegment:
    """The genes of ONE drone (one segment of the chromosome)."""

    def __init__(self, base: int, waypoints: np.ndarray, speeds: np.ndarray,
                 powers: np.ndarray, sensor_on: np.ndarray) -> None:
        self.base = int(base)                                      # integer gene
        self.waypoints = np.asarray(waypoints, dtype=float).reshape(-1, 2)  # K x 2 real genes
        self.speeds = np.asarray(speeds, dtype=float)              # K real genes
        self.powers = np.asarray(powers, dtype=float)              # K real genes
        self.sensor_on = np.asarray(sensor_on, dtype=bool)         # K binary genes

    @property
    def n_waypoints(self) -> int:
        return len(self.waypoints)

    def copy(self) -> "DroneSegment":
        return DroneSegment(self.base, self.waypoints.copy(), self.speeds.copy(),
                            self.powers.copy(), self.sensor_on.copy())

    def route(self, env: Environment) -> np.ndarray:
        """All points the drone visits, including take-off and landing at its base."""
        base_xy = env.base_stations[self.base]
        return np.vstack([base_xy, self.waypoints, base_xy])

    @staticmethod
    def random(env: Environment, rng: np.random.Generator) -> "DroneSegment":
        """Create a random but valid drone segment.

        Waypoints are drawn from OPEN cells only, so a new random solution
        never starts inside a no-fly zone (legs may still cross one; the
        repair operator fixes that later).
        """
        k = int(rng.integers(cfg.K_INIT_MIN, cfg.K_INIT_MAX + 1))
        open_cells = np.argwhere(~env.blocked_margin)              # rows of (y, x)
        chosen = open_cells[rng.integers(0, len(open_cells), size=k)]
        waypoints = np.column_stack([chosen[:, 1], chosen[:, 0]]) + rng.random((k, 2))
        return DroneSegment(
            base=rng.integers(0, len(env.base_stations)),
            waypoints=waypoints,
            speeds=rng.uniform(cfg.V_MIN, cfg.V_MAX, size=k),
            powers=rng.random(k),
            sensor_on=rng.random(k) < cfg.INIT_SENSOR_ON_PROB,
        )


class Chromosome:
    """A full fleet plan = a list of DroneSegment objects (one per drone).

    After evaluation the chromosome also remembers its raw objective values,
    its constraint violation and its fitness, so we never evaluate the same
    unchanged individual twice (e.g. elites copied to the next generation).
    """

    def __init__(self, segments: list[DroneSegment]) -> None:
        self.segments = segments
        self.objectives: dict | None = None   # filled in by FitnessEvaluator
        self.violation: float = 0.0
        self.fitness: float = -np.inf

    @staticmethod
    def random(env: Environment, rng: np.random.Generator) -> "Chromosome":
        return Chromosome([DroneSegment.random(env, rng) for _ in range(cfg.N_DRONES)])

    def copy(self) -> "Chromosome":
        """Deep copy (genes + cached evaluation)."""
        clone = Chromosome([s.copy() for s in self.segments])
        clone.objectives = None if self.objectives is None else dict(self.objectives)
        clone.violation = self.violation
        clone.fitness = self.fitness
        return clone

    def forget_fitness(self) -> None:
        """Call after changing any gene: the old fitness is no longer valid."""
        self.objectives = None
        self.violation = 0.0
        self.fitness = -np.inf

    def n_genes(self) -> int:
        """Total number of genes (1 base + 5 genes per waypoint, per drone)."""
        return sum(1 + 5 * s.n_waypoints for s in self.segments)

    def is_within_bounds(self, env: Environment) -> bool:
        """Check every gene is inside its allowed range (used by the tests)."""
        for s in self.segments:
            ok = (0 <= s.base < len(env.base_stations)
                  and np.all(s.waypoints >= 0) and np.all(s.waypoints < env.size)
                  and np.all(s.speeds >= cfg.V_MIN - 1e-9) and np.all(s.speeds <= cfg.V_MAX + 1e-9)
                  and np.all(s.powers >= 0) and np.all(s.powers <= 1)
                  and len(s.speeds) == len(s.powers) == len(s.sensor_on) == s.n_waypoints)
            if not ok:
                return False
        return True

    def describe(self) -> list[dict]:
        """A JSON-friendly summary of the plan (saved in run_summary.json)."""
        return [{
            "drone": i + 1,
            "base": s.base,
            "waypoints": np.round(s.waypoints, 2).tolist(),
            "speeds_mps": np.round(s.speeds, 2).tolist(),
            "sensor_power": np.round(s.powers, 2).tolist(),
            "sensor_on": s.sensor_on.astype(int).tolist(),
        } for i, s in enumerate(self.segments)]


def path_features(chrom: Chromosome, env: Environment, n_points: int = 10) -> np.ndarray:
    """Turn a chromosome into a fixed-length vector to measure diversity.

    Chromosomes have different lengths, so we cannot compare genes one by
    one. Instead each drone route is resampled to ``n_points`` points spaced
    evenly along the route, divided by the grid size (so the values are
    0..1), and followed by the drone's mean speed, power and sensor-on share.
    """
    parts = []
    for seg in chrom.segments:
        route = seg.route(env)
        step = np.hypot(*np.diff(route, axis=0).T)
        dist = np.concatenate([[0.0], np.cumsum(step)])
        where = np.linspace(0.0, dist[-1], n_points)
        xs = np.interp(where, dist, route[:, 0]) / env.size
        ys = np.interp(where, dist, route[:, 1]) / env.size
        speed = (seg.speeds.mean() - cfg.V_MIN) / (cfg.V_MAX - cfg.V_MIN)
        parts.append(np.concatenate([xs, ys, [speed, seg.powers.mean(), seg.sensor_on.mean()]]))
    return np.concatenate(parts)
