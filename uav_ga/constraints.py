"""
Constraint handling: the REPAIR operator.

Hard constraints of the problem:
    1. every gene must be inside its range (grid bounds, speed limits, ...)
    2. no waypoint and no flight leg may enter a no-fly zone
    3. each drone must stay within MAX_RANGE_M and its BATTERY_J

Two ways to handle them are implemented (chosen with cfg.CONSTRAINT_MODE):
    * "penalty" - only clip genes to their bounds; every remaining violation
                  is punished by the dynamic penalty in fitness.py
    * "repair"  - actively fix the chromosome (this file), no penalty
    * "both"    - repair first, then penalise anything repair could not fix

The repaired genes are written back into the chromosome ("Lamarckian"
repair), so children inherit the feasible version of their parents.
"""

import numpy as np

from uav_ga import config as cfg
from uav_ga.chromosome import Chromosome, DroneSegment
from uav_ga.environment import Environment
from uav_ga.fitness import flight_power, simulate_flight


def clip_genes(chrom: Chromosome, env: Environment) -> None:
    """Constraint 1: force every gene back into its allowed range."""
    for seg in chrom.segments:
        seg.base = int(np.clip(seg.base, 0, len(env.base_stations) - 1))
        np.clip(seg.waypoints, 0.0, env.size - 1e-6, out=seg.waypoints)
        np.clip(seg.speeds, cfg.V_MIN, cfg.V_MAX, out=seg.speeds)
        np.clip(seg.powers, 0.0, 1.0, out=seg.powers)


def move_waypoints_out_of_nofly(seg: DroneSegment, env: Environment) -> None:
    """Constraint 2a: a waypoint inside a no-fly zone jumps to the nearest free cell.

    The nearest free cell of every cell was found once with a BFS when the
    environment was built (env.nearest_free), so this is a quick lookup.
    """
    for k, (x, y) in enumerate(seg.waypoints):
        if not env.is_free(np.array([x, y])):
            fx, fy = env.nearest_free[int(y), int(x)]
            seg.waypoints[k] = (fx + 0.5, fy + 0.5)


def find_detour(a: np.ndarray, b: np.ndarray, env: Environment, depth: int = 0) -> list[np.ndarray] | None:
    """Return extra points that lead from a to b around the no-fly zones.

    Step 1 (cheap): try ONE extra point to the left or right of the middle of
            the leg, moving further out (3, 6, 9 ... cells) until both new
            legs are clear.
    Step 2 (divide and conquer): if no single point works, take the closest
            free side point that clears at least one half of the leg and
            solve the two halves again with the same method (max 3 levels).
    Step 3 (fallback): run A* on the grid, then remove unnecessary points
            ("string pulling": keep jumping to the furthest point that can be
            reached in a straight line).
    Returns [] if the leg is already clear and None if no detour exists.
    """
    if env.segment_is_clear(a, b):
        return []
    direction = b - a
    length = float(np.hypot(*direction))
    if length > 1e-9:
        normal = np.array([-direction[1], direction[0]]) / length   # 90 degrees to the leg
        middle = (a + b) / 2
        half_way_point = None
        for offset in range(3, 46, 3):
            for side in (1, -1):
                c = middle + side * offset * normal
                if not env.is_free(c):
                    continue
                first_ok, second_ok = env.segment_is_clear(a, c), env.segment_is_clear(c, b)
                if first_ok and second_ok:
                    return [c]                                       # step 1 worked
                if half_way_point is None and (first_ok or second_ok):
                    half_way_point = c
        if half_way_point is not None and depth < 3:                 # step 2
            first = find_detour(a, half_way_point, env, depth + 1)
            second = find_detour(half_way_point, b, env, depth + 1)
            if first is not None and second is not None:
                return first + [half_way_point] + second

    cells = env.astar((int(a[0]), int(a[1])), (int(b[0]), int(b[1])))   # step 3
    if cells is None:
        return None
    points = [a] + [np.array([cx + 0.5, cy + 0.5]) for cx, cy in cells[1:-1]] + [b]
    kept, i = [], 0
    while i < len(points) - 1:
        j = len(points) - 1
        while j > i + 1 and not env.segment_is_clear(points[i], points[j]):
            j -= 1
        kept.append(points[j])
        i = j
    return kept[:-1]          # the last kept point is b itself


def fix_crossing_legs(seg: DroneSegment, env: Environment) -> None:
    """Constraint 2b: insert detour waypoints into legs that cross a no-fly zone.

    A detour point inherits the speed / power / sensor bit of the leg it
    splits, so the drone's sensor schedule does not change. Detour points
    on the return-to-base leg use the return speed with the sensor off.
    """
    base_xy = env.base_stations[seg.base]
    new_wp, new_v, new_p, new_on = [], [], [], []
    previous = base_xy
    for k in range(seg.n_waypoints + 1):            # +1 = the return leg
        is_return = k == seg.n_waypoints
        target = base_xy if is_return else seg.waypoints[k]
        speed = cfg.V_RETURN if is_return else seg.speeds[k]
        power = 0.0 if is_return else seg.powers[k]
        sensor = False if is_return else seg.sensor_on[k]
        for point in find_detour(previous, target, env) or []:
            new_wp.append(point); new_v.append(speed); new_p.append(power); new_on.append(sensor)
        if not is_return:
            new_wp.append(target); new_v.append(speed); new_p.append(power); new_on.append(sensor)
        previous = target
    seg.waypoints = np.array(new_wp).reshape(-1, 2)
    seg.speeds, seg.powers, seg.sensor_on = np.array(new_v), np.array(new_p), np.array(new_on, dtype=bool)


def trim_to_limits(seg: DroneSegment, env: Environment) -> None:
    """Constraint 3: drop waypoints from the END of the route until the drone
    can fly the route AND come back home within its range and battery.

    1. If the route is already within the limits, nothing happens.
    2. Otherwise we ESTIMATE how many waypoints can be kept: for every k we
       add the cost of flying to waypoint k and a pessimistic straight flight
       home (full wind). This avoids re-simulating the route k times.
    3. We then check the shortened route exactly (the way home may need a
       detour around a no-fly zone) and drop one more waypoint if needed.
    """
    flight = simulate_flight(Chromosome([seg]), env)
    if flight["length_m"][0] <= cfg.MAX_RANGE_M and flight["energy_j"][0] <= cfg.BATTERY_J:
        return
    base_xy = env.base_stations[seg.base]
    distance_so_far = np.cumsum(flight["leg_length_m"][:-1])     # to waypoint 1, 2, ..., K
    energy_so_far = np.cumsum(flight["leg_energy_j"][:-1])
    home_m = np.hypot(*(seg.waypoints - base_xy).T) * env.cell_size_m
    home_j = home_m / cfg.V_RETURN * flight_power(cfg.V_RETURN) * (1 + cfg.WIND_POWER_FACTOR)
    fits = (distance_so_far + home_m <= cfg.MAX_RANGE_M) & (energy_so_far + home_j <= cfg.BATTERY_J)
    keep = int(np.flatnonzero(fits).max()) + 1 if fits.any() else 1

    while True:
        trial = DroneSegment(seg.base, seg.waypoints[:keep], seg.speeds[:keep],
                             seg.powers[:keep], seg.sensor_on[:keep])
        for point in find_detour(trial.waypoints[-1], base_xy, env) or []:
            trial.waypoints = np.vstack([trial.waypoints, point])
            trial.speeds = np.append(trial.speeds, cfg.V_RETURN)
            trial.powers = np.append(trial.powers, 0.0)
            trial.sensor_on = np.append(trial.sensor_on, False)
        check = simulate_flight(Chromosome([trial]), env)
        if (check["length_m"][0] <= cfg.MAX_RANGE_M and check["energy_j"][0] <= cfg.BATTERY_J) or keep <= 1:
            break
        keep -= 1
    seg.waypoints, seg.speeds, seg.powers, seg.sensor_on = (
        trial.waypoints, trial.speeds, trial.powers, trial.sensor_on)


def repair(chrom: Chromosome, env: Environment, mode: str = cfg.CONSTRAINT_MODE) -> Chromosome:
    """Apply the constraint handling chosen by ``mode`` to ``chrom`` (in place)."""
    clip_genes(chrom, env)
    if mode in ("repair", "both"):
        for seg in chrom.segments:
            move_waypoints_out_of_nofly(seg, env)
            fix_crossing_legs(seg, env)
            trim_to_limits(seg, env)
    chrom.forget_fitness()
    return chrom
