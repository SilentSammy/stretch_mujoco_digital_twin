try:
    from rplidar import RPLidar
    from stretch_body.robot import Robot
except ImportError:
    from stretch_mujoco_api.rplidar import RPLidar
    from stretch_mujoco_api.robot import Robot

robot = Robot()
robot.startup()

try:
    lidar = RPLidar("/dev/hello-lrf")
    try:
        for scan in lidar.iter_scans():
            print([(angle, distance / 1000) for _, angle, distance in scan])
    finally:
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
finally:
    robot.stop()
