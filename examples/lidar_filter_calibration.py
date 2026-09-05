import time

import numpy as np

from stretch_tools import NormVelController

try:
    from rplidar import RPLidar
    import stretch_body.robot as robot
except ImportError:
    from stretch_mujoco_api.rplidar import RPLidar
    import stretch_mujoco_api.robot as robot


ROTATION_SPEED = 0.33
WARMUP_SECONDS = 1.0
CALIBRATION_SECONDS = 14.0
MAX_SELF_DISTANCE = 0.5
ANGLE_MARGIN = 2
MAX_MEDIAN_DEVIATION = 0.02
MIN_OBSERVATION_RATIO = 0.5


def bin_scan(scan):
    bins = np.full(360, np.nan)
    measurements = [[] for _ in range(360)]

    for _, angle, distance_mm in scan:
        measurements[round(angle) % 360].append(distance_mm / 1000)

    for angle, distances in enumerate(measurements):
        if distances:
            bins[angle] = np.median(distances)

    return bins


def find_fixed_points(scans):
    if not scans:
        return []

    fixed_points = []
    required_observations = len(scans) * MIN_OBSERVATION_RATIO
    scans = np.asarray(scans)

    for angle in range(360):
        neighboring_angles = [
            (angle + offset) % 360
            for offset in range(-ANGLE_MARGIN, ANGLE_MARGIN + 1)
        ]
        nearby = scans[:, neighboring_angles]
        samples = []
        for row in nearby:
            row = row[np.isfinite(row) & (row <= MAX_SELF_DISTANCE)]
            if len(row):
                samples.append(np.median(row))

        if len(samples) < required_observations:
            continue

        distance = float(np.median(samples))
        deviation = np.median(np.abs(np.asarray(samples) - distance))
        if deviation <= MAX_MEDIAN_DEVIATION:
            fixed_points.append((angle, distance))

    return fixed_points


def main():
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()

    controller = NormVelController(stretch, safe_base_mode=True)
    lidar = RPLidar("/dev/hello-lrf")
    scans = []
    started = time.monotonic()

    try:
        for scan in lidar.iter_scans():
            elapsed = time.monotonic() - started
            controller.set_command({"base_counterclockwise": ROTATION_SPEED})

            if elapsed >= WARMUP_SECONDS:
                scans.append(bin_scan(scan))
            if elapsed >= WARMUP_SECONDS + CALIBRATION_SECONDS:
                break
    finally:
        controller.set_command({"base_counterclockwise": 0.0})
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        stretch.stop()

    fixed_points = find_fixed_points(scans)
    print("MAST_PROFILE = [")
    for angle, distance in fixed_points:
        print(f"    ({angle}, {distance:.3f}),")
    print("]")


if __name__ == "__main__":
    main()
