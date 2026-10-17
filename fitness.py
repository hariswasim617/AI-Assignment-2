"""
Fitness function and constraint handling.

    F = 0.35*Coverage + 0.25*Detection - 0.15*Energy - 0.10*Time - 0.15*Hazard - 2*Penalty

All five objectives are scaled to 0..1 so the weights can be compared fairly.
Constraints:
    * repair  : a waypoint inside a no-fly zone is moved to the nearest open cell
    * penalty : flying THROUGH a no-fly zone, or flying further than MAX_RANGE
"""
import numpy as np

import config as cfg


def sensor_radius(power):
    """Sensor footprint radius in cells: 1 cell at power 0, 4 cells at power 1."""
    return 1 + 3 * power


def repair(chrom, env):
    """Clip the genes to their ranges and move waypoints out of no-fly zones."""
    chrom.clip()
    for d in range(cfg.N_DRONES):
        for k in range(cfg.N_WAYPOINTS):
            x, y = chrom.genes[d, k, 0], chrom.genes[d, k, 1]
            if env.is_blocked(x, y):
                chrom.genes[d, k, 0:2] = env.nearest_free(x, y)
    return chrom


def fly(chrom):
    """Turn the chromosome into a list of straight legs.

    Returns, for every leg: start point, end point, speed and sensor radius
    (radius 0 = sensor off), and which drone flies it.
    Leg k of a drone ends at waypoint k and uses that waypoint's speed and
    sensor power. The last leg (back to base) is flown at full speed with the
    sensor off.
    """
    starts, ends, speeds, radii, drone_of_leg = [], [], [], [], []
    for d in range(cfg.N_DRONES):
        route = chrom.route(d)
        for k in range(len(route) - 1):
            starts.append(route[k])
            ends.append(route[k + 1])
            if k < cfg.N_WAYPOINTS:
                speeds.append(chrom.genes[d, k, 2])
                radii.append(sensor_radius(chrom.genes[d, k, 3]))
            else:
                speeds.append(cfg.SPEED_MAX)
                radii.append(0.0)
            drone_of_leg.append(d)
    return np.array(starts), np.array(ends), np.array(speeds), np.array(radii), np.array(drone_of_leg)


def distance_to_legs(px, py, starts, ends):
    """Distance from points (px, py) to every leg (a table: points x legs).

    Standard point-to-segment distance: project the point onto the leg,
    keep the projection inside the leg (0 <= t <= 1) and measure.
    """
    ax, ay = starts[:, 0][None, :], starts[:, 1][None, :]
    bx, by = ends[:, 0][None, :], ends[:, 1][None, :]
    px, py = px[:, None], py[:, None]
    abx, aby = bx - ax, by - ay
    t = np.clip(((px - ax) * abx + (py - ay) * aby) / (abx ** 2 + aby ** 2 + 1e-9), 0, 1)
    return np.hypot(px - (ax + t * abx), py - (ay + t * aby))


def coverage_map(chrom, env):
    """50x50 True/False map: True if the cell is inside a sensor footprint."""
    starts, ends, _, radii, _ = fly(chrom)
    dist = distance_to_legs(env.cell_x, env.cell_y, starts, ends)
    return (dist <= radii[None, :]).any(axis=1).reshape(env.size, env.size)


def points_along_legs(starts, ends):
    """Sample one point per cell flown along every leg (used for hazard / no-fly checks)."""
    xs, ys = [], []
    for a, b in zip(starts, ends):
        n = max(1, int(np.hypot(*(b - a))))
        t = np.linspace(0, 1, n + 1)[1:]
        xs.append(a[0] + t * (b[0] - a[0]))
        ys.append(a[1] + t * (b[1] - a[1]))
    return np.concatenate(xs), np.concatenate(ys)


def evaluate(chrom, env):
    """Compute all objectives and the final fitness of one chromosome."""
    starts, ends, speeds, radii, drone = fly(chrom)
    lengths = np.hypot(*(ends - starts).T)
    times = lengths / speeds
    sensor_power = (radii - 1) / 3 * (radii > 0)          # back from radius to power (0 if off)

    # MAXIMISE 1: priority-weighted coverage of the map
    covered = coverage_map(chrom, env)
    coverage = (env.priority * covered).sum() / env.priority.sum()

    # MAXIMISE 2: share of targets that came inside a sensor footprint
    t_dist = distance_to_legs(env.targets[:, 0], env.targets[:, 1], starts, ends)
    detection = (t_dist <= radii[None, :]).any(axis=1).mean()

    # MINIMISE 1: energy. Flying faster costs more per cell; the sensor costs energy per time step.
    energy = (lengths * (1 + 0.3 * speeds) + 1.5 * sensor_power * times).sum()
    max_energy = cfg.N_DRONES * cfg.MAX_RANGE * (1 + 0.3 * cfg.SPEED_MAX + 1.5 / cfg.SPEED_MIN)
    energy_n = min(1.0, energy / max_energy)

    # MINIMISE 2: mission time = time of the slowest drone
    drone_time = np.bincount(drone, times, minlength=cfg.N_DRONES)
    time_n = min(1.0, drone_time.max() / (cfg.MAX_RANGE / cfg.SPEED_MIN))

    # MINIMISE 3: hazard exposure = average hazard under the flown path
    xs, ys = points_along_legs(starts, ends)
    xi = np.clip(xs.astype(int), 0, env.size - 1)
    yi = np.clip(ys.astype(int), 0, env.size - 1)
    hazard = env.hazard[xi, yi].mean()

    # PENALTY: share of the path inside no-fly zones (x10) + flying further than allowed
    nofly = env.no_fly[xi, yi].mean()
    drone_distance = np.bincount(drone, lengths, minlength=cfg.N_DRONES)
    over_range = np.maximum(0, drone_distance / cfg.MAX_RANGE - 1).sum()
    penalty = 10 * nofly + over_range

    fitness = (cfg.W_COVERAGE * coverage + cfg.W_DETECTION * detection
               - cfg.W_ENERGY * energy_n - cfg.W_TIME * time_n - cfg.W_HAZARD * hazard
               - cfg.PENALTY_WEIGHT * penalty)

    chrom.fitness = fitness
    chrom.details = {"coverage": coverage, "detection": detection, "energy": energy_n,
                     "time": time_n, "hazard": hazard, "penalty": penalty}
    return fitness
