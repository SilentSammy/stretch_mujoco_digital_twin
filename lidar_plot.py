from stretch_tools import NormVelController, TeleopProvider
from stretch_tools.lidar_plotter import LidarPlotter

try:
    from rplidar import RPLidar
    import stretch_body.robot as robot
except ImportError:
    from stretch_mujoco_api.rplidar import RPLidar
    import stretch_mujoco_api.robot as robot


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
            plotter.update(scan)
            controller.set_command(teleop.get_manual_override({}))
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
