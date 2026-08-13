import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot

if __name__ == "__main__":
    robot = robot.Robot()
    robot.startup()
    robot.enable_collision_mgmt()
    try:
        def compact_status():
            status = robot.get_status()
            compact = {}

            for name in ("arm",):
                joint = status[name]
                motor = joint["motor"]
                compact[name] = {
                    "timestamp_pc": joint["timestamp_pc"],
                    "pos": joint["pos"],
                    "vel": joint["vel"],
                    "motor": {
                        "timestamp": motor["timestamp"],
                        "near_pos_setpoint": motor["near_pos_setpoint"],
                        "is_moving": motor["is_moving"],
                        "is_moving_filtered": motor["is_moving_filtered"],
                        "is_mg_moving": motor["is_mg_moving"],
                    },
                }

            return compact

        robot.arm.move_to(0.2)
        robot.push_command()
        robot.wait_command()
        print("initial", compact_status())

        robot.arm.move_by(0.2)
        robot.push_command()

        for sample in range(10):
            print(sample, compact_status())
            time.sleep(0.1)

        robot.wait_command()
        print("settled", compact_status())

    finally:
        robot.stop()  # Stop the robot and clean up
