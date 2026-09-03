import numpy as np

from stretch_tools import (
    LidarPlotter,
    NormVelController,
    TeleopProvider,
    filter_mast_points,
)
from stretch_tools.norm_vel_ctrl import merge_proportional

try:
    from rplidar import RPLidar
    import stretch_body.robot as robot
except ImportError:
    from stretch_mujoco_api.rplidar import RPLidar
    import stretch_mujoco_api.robot as robot


HALF_BASE_WIDTH = 0.175
FRONT_DETECTION_DISTANCE = 0.3
REAR_DETECTION_DISTANCE = 0.55
REAR_FULL_AVOIDANCE_DISTANCE = 0.15


def get_obstacle_avoidance(scan):
    angles = np.radians([angle for _, angle, _ in scan])
    distances = np.array([distance for _, _, distance in scan]) / 1000
    x = distances * np.cos(angles)
    y = distances * np.sin(angles)

    front_obstacles = (
        (np.abs(y) <= HALF_BASE_WIDTH)
        & (x > 0)
        & (x <= FRONT_DETECTION_DISTANCE)
    )
    rear_obstacles = (
        (np.abs(y) <= HALF_BASE_WIDTH)
        & (x <= 0)
        & (x >= -REAR_DETECTION_DISTANCE)
    )

    contributions = []
    if front_obstacles.any():
        index = np.flatnonzero(front_obstacles)[np.argmin(x[front_obstacles])]
        contribution = -(1 - x[index] / FRONT_DETECTION_DISTANCE)
        contributions.append(contribution)

    if rear_obstacles.any():
        index = np.flatnonzero(rear_obstacles)[np.argmax(x[rear_obstacles])]
        closest = abs(x[index])
        authority = (REAR_DETECTION_DISTANCE - closest) / (
            REAR_DETECTION_DISTANCE - REAR_FULL_AVOIDANCE_DISTANCE
        )
        contributions.append(min(1.0, authority))

    if not contributions:
        return {}

    return {"base_forward": max(contributions, key=abs)}


def main():
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()

    controller = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)
    lidar = RPLidar("/dev/hello-lrf")
    plotter = LidarPlotter()

    try:
        for scan in lidar.iter_scans():
            scan = filter_mast_points(scan)
            avoidance = get_obstacle_avoidance(scan)
            teleop_command = teleop.get_normalized_velocities()
            command = merge_proportional(avoidance, teleop_command)

            controller.set_command(command)
            plotter.update(scan)
    except KeyboardInterrupt:
        pass
    finally:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        plotter.close()
        stretch.stop()


if __name__ == "__main__":
    main()
