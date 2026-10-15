"""
Flight simulation, UAV energy model and the multi-objective fitness function.

    F = w1*Coverage + w2*Detection - w3*Energy - w4*FlightTime - w5*Hazard - Penalty

Every term is normalised to 0..1 so that the weights w1..w5 really express
how important each goal is (otherwise a term measured in joules would
dominate a term measured as a fraction).

    Coverage    priority-weighted share of the map seen by an ACTIVE sensor
    Detection   expected share of targets detected
    Energy      fleet energy used / fleet battery capacity
    FlightTime  mission makespan (slowest drone) / longest possible flight
    Hazard      time-weighted average hazard intensity along all flights
    Penalty     dynamic penalty for constraint violations (see constraints.py)
"""

import numpy as np

from uav_ga import config as cfg
from uav_ga.chromosome import Chromosome
from uav_ga.environment import Environment

# The longest a single feasible flight can last: max range at minimum speed.
REFERENCE_TIME_S = cfg.MAX_RANGE_M / cfg.V_MIN


# ---------------------------------------------------------------------------
# UAV model
# ---------------------------------------------------------------------------
def flight_power(speed: np.ndarray) -> np.ndarray:
    """Propulsion power (W) of a rotary-wing drone flying at ``speed`` (m/s).

    Model of Zeng, Xu & Zhang (2019). It has three parts:
      * blade profile power - grows slowly with speed
      * induced power       - needed to stay in the air, DROPS as speed grows
      * parasite power      - air drag on the body, grows with speed^3
    Because of this U-shape, hovering or flying very slowly is NOT the
    cheapest option. With these constants the energy per metre (power / speed)
    keeps falling up to about 18 m/s, so inside our speed limits a faster
    drone always spends less energy per metre (but detects targets worse).
    """
    v = np.asarray(speed, dtype=float)
    blade = cfg.BLADE_PROFILE_POWER * (1 + 3 * v ** 2 / cfg.ROTOR_TIP_SPEED ** 2)
    v0 = cfg.MEAN_ROTOR_VELOCITY
    induced = cfg.INDUCED_POWER * np.sqrt(np.sqrt(1 + v ** 4 / (4 * v0 ** 4)) - v ** 2 / (2 * v0 ** 2))
    parasite = 0.5 * cfg.FUSELAGE_DRAG_RATIO * cfg.AIR_DENSITY * cfg.ROTOR_SOLIDITY * cfg.ROTOR_DISC_AREA * v ** 3
    return blade + induced + parasite


def sensor_power_w(power_gene: np.ndarray) -> np.ndarray:
    """Electrical power drawn by the sensor for a power gene in 0..1."""
    return cfg.SENSOR_POWER_MIN_W + (cfg.SENSOR_POWER_MAX_W - cfg.SENSOR_POWER_MIN_W) * power_gene


def sensor_radius(power_gene: np.ndarray) -> np.ndarray:
    """Footprint radius in cells: a stronger sensor sees further."""
    return cfg.SENSOR_RADIUS_MIN + (cfg.SENSOR_RADIUS_MAX - cfg.SENSOR_RADIUS_MIN) * power_gene


# ---------------------------------------------------------------------------
# Flight simulation
# ---------------------------------------------------------------------------
def build_legs(chrom: Chromosome, env: Environment) -> dict:
    """Flatten the whole fleet into one table of straight legs.

    Leg k of a drone ends at waypoint k and uses that waypoint's speed,
    power and sensor bit. The extra last leg returns to base at V_RETURN
    with the sensor off.
    """
    starts, ends, speeds, powers, active, drone = [], [], [], [], [], []
    for d, seg in enumerate(chrom.segments):
        route = seg.route(env)
        starts.append(route[:-1])
        ends.append(route[1:])
        speeds.append(np.append(seg.speeds, cfg.V_RETURN))
        powers.append(np.append(seg.powers, 0.0))
        active.append(np.append(seg.sensor_on, False))
        drone.append(np.full(len(route) - 1, d))
    return {"start": np.vstack(starts), "end": np.vstack(ends),
            "speed": np.concatenate(speeds), "power": np.concatenate(powers),
            "active": np.concatenate(active), "drone": np.concatenate(drone)}


def simulate_flight(chrom: Chromosome, env: Environment) -> dict:
    """Fly every leg and compute distance, time, energy and hazard dose.

    Returns per-drone totals (arrays of length N_DRONES) plus the leg table.
    """
    legs = build_legs(chrom, env)
    probe = env.sample_legs(legs["start"], legs["end"])
    length_m = probe["length"] * env.cell_size_m
    time_s = length_m / legs["speed"]
    # wind makes the motors work harder: power x (1 + 0.6 * mean wind)
    propulsion_w = flight_power(legs["speed"]) * (1 + cfg.WIND_POWER_FACTOR * probe["mean_wind"])
    sensor_w = np.where(legs["active"], sensor_power_w(legs["power"]), 0.0)
    energy_j = (propulsion_w + sensor_w) * time_s
    hazard_dose = probe["mean_hazard"] * time_s   # intensity x seconds spent

    n = len(chrom.segments)
    per_drone = lambda values: np.bincount(legs["drone"], values, minlength=n)
    return {
        "legs": legs,
        "leg_length_m": length_m,
        "leg_energy_j": energy_j,
        "length_m": per_drone(length_m),
        "time_s": per_drone(time_s),
        "energy_j": per_drone(energy_j),
        "hazard_dose": per_drone(hazard_dose),
        "blocked_samples": int(probe["blocked_samples"].sum()),
        "total_samples": int(probe["samples"].sum()),
    }


def point_to_segment_distance(px, py, ax, ay, bx, by):
    """Shortest distance from point(s) (px, py) to the segment A-B.

    We project the point onto the line, clamp the projection to the segment
    (t in 0..1) and measure the distance to that closest point.
    """
    abx, aby = bx - ax, by - ay
    denom = abx * abx + aby * aby + 1e-12
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / denom, 0.0, 1.0)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def coverage_map(chrom: Chromosome, env: Environment, legs: dict | None = None) -> np.ndarray:
    """Boolean map[y, x]: True where some active sensor footprint covered the cell.

    A cell is covered if its centre is within the sensor radius of a leg
    flown with the sensor ON. To save time we only check the cells inside
    the bounding box of each leg (grown by the radius).
    """
    if legs is None:
        legs = build_legs(chrom, env)
    covered = np.zeros((env.size, env.size), dtype=bool)
    centres = env.cell_centres_x
    for i in np.flatnonzero(legs["active"]):
        (ax, ay), (bx, by) = legs["start"][i], legs["end"][i]
        r = float(sensor_radius(legs["power"][i]))
        x0, x1 = int(max(0, min(ax, bx) - r)), int(min(env.size, max(ax, bx) + r + 1))
        y0, y1 = int(max(0, min(ay, by) - r)), int(min(env.size, max(ay, by) + r + 1))
        px = centres[x0:x1][None, :]
        py = centres[y0:y1][:, None]
        covered[y0:y1, x0:x1] |= point_to_segment_distance(px, py, ax, ay, bx, by) <= r
    return covered


def detection_probabilities(legs: dict, env: Environment) -> np.ndarray:
    """Probability that each target is detected by at least one active leg.

    For one leg:  P = quality(power) * speed_factor(speed) * max(0, 1 - (d / r)^2)
      * quality      0.4 .. 1.0, a stronger sensor is more reliable
      * speed_factor 1.0 at V_MIN .. 0.5 at V_MAX (motion blur)
      * distance     targets near the centre line are easier to spot
    Several legs act like independent tries:  P_total = 1 - prod(1 - P_leg).
    """
    active = legs["active"]
    if not active.any():
        return np.zeros(len(env.targets))
    a, b = legs["start"][active], legs["end"][active]
    power, speed = legs["power"][active], legs["speed"][active]
    tx, ty = env.targets[:, 0][None, :], env.targets[:, 1][None, :]
    dist = point_to_segment_distance(tx, ty, a[:, :1], a[:, 1:], b[:, :1], b[:, 1:])
    radius = sensor_radius(power)[:, None]
    quality = (0.4 + 0.6 * power)[:, None]
    speed_factor = (1 - cfg.DETECTION_SPEED_FACTOR * (speed - cfg.V_MIN) / (cfg.V_MAX - cfg.V_MIN))[:, None]
    p_leg = quality * speed_factor * np.clip(1 - (dist / radius) ** 2, 0.0, 1.0)
    return 1.0 - np.prod(1.0 - p_leg, axis=0)


# ---------------------------------------------------------------------------
# Fitness evaluator
# ---------------------------------------------------------------------------
class FitnessEvaluator:
    """Computes the objectives and the scalar fitness of chromosomes.

    It also counts how many evaluations were made, so the random-search
    baseline can be given exactly the same budget as the GA.
    """

    def __init__(self, env: Environment, total_generations: int = cfg.GENERATIONS,
                 constraint_mode: str = cfg.CONSTRAINT_MODE) -> None:
        self.env = env
        self.total_generations = max(1, total_generations)
        self.constraint_mode = constraint_mode
        self.n_evaluations = 0
        self.weights = np.array([cfg.W_COVERAGE, cfg.W_DETECTION, cfg.W_ENERGY, cfg.W_TIME, cfg.W_HAZARD])

    def objectives(self, chrom: Chromosome) -> dict:
        """Simulate the plan and return all raw and normalised objective values."""
        flight = simulate_flight(chrom, self.env)
        legs = flight["legs"]
        covered = coverage_map(chrom, self.env, legs)
        p_detect = detection_probabilities(legs, self.env)
        n = len(chrom.segments)
        total_time = flight["time_s"].sum()

        # --- constraint violations (0 means feasible) ---
        nofly_fraction = flight["blocked_samples"] / max(1, flight["total_samples"])
        range_overrun = np.maximum(0, flight["length_m"] / cfg.MAX_RANGE_M - 1).sum() / n
        battery_overrun = np.maximum(0, flight["energy_j"] / cfg.BATTERY_J - 1).sum() / n

        high_priority = self.env.priority >= 0.5
        return {
            # normalised objectives (0..1)
            "coverage": float((self.env.priority * covered).sum() / self.env.priority_total),
            "detection": float(p_detect.mean()),
            "energy": float(flight["energy_j"].sum() / (n * cfg.BATTERY_J)),
            "time": float(flight["time_s"].max() / REFERENCE_TIME_S),
            "hazard": float(flight["hazard_dose"].sum() / max(total_time, 1e-9)),
            # raw values, logged for the report
            "high_priority_coverage": float(covered[high_priority].mean()),
            "area_covered_frac": float(covered.mean()),
            "targets_detected": int((p_detect >= 0.5).sum()),
            "energy_kj": float(flight["energy_j"].sum() / 1000),
            "makespan_s": float(flight["time_s"].max()),
            "distance_km": float(flight["length_m"].sum() / 1000),
            "nofly_fraction": float(nofly_fraction),
            "range_overrun": float(range_overrun),
            "battery_overrun": float(battery_overrun),
            "n_waypoints": int(sum(s.n_waypoints for s in chrom.segments)),
        }

    def penalty_factor(self, generation: int) -> float:
        """Dynamic penalty multiplier: small early (explore), large late (be feasible).

        lambda(g) = PENALTY_START * (1 + PENALTY_GROWTH * g / G)
        In pure "repair" mode the penalty is switched off (lambda = 0).
        """
        if self.constraint_mode == "repair":
            return 0.0
        progress = min(1.0, generation / self.total_generations)
        return cfg.PENALTY_START * (1 + cfg.PENALTY_GROWTH * progress)

    @staticmethod
    def violation(obj: dict) -> float:
        """One number that measures how badly the hard constraints are broken."""
        return cfg.NOFLY_WEIGHT * obj["nofly_fraction"] + obj["range_overrun"] + obj["battery_overrun"]

    def score(self, obj: dict, generation: int) -> float:
        """Turn the objective values into the single fitness number F."""
        terms = np.array([obj["coverage"], obj["detection"], obj["energy"], obj["time"], obj["hazard"]])
        signs = np.array([1, 1, -1, -1, -1])
        weighted = float(np.sum(signs * self.weights * terms))
        return weighted - self.penalty_factor(generation) * self.violation(obj)

    def evaluate(self, chrom: Chromosome, generation: int) -> float:
        """Evaluate a chromosome (only if it is new) and store the results on it."""
        if chrom.objectives is None:
            chrom.objectives = self.objectives(chrom)
            self.n_evaluations += 1
        chrom.violation = self.violation(chrom.objectives)
        chrom.fitness = self.score(chrom.objectives, generation)
        return chrom.fitness
