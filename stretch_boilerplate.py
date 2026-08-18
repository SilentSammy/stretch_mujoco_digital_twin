try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot

if __name__ == "__main__":
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()
    try:
        while True:
            pass  # Your code here
    finally:
        stretch.stop()
