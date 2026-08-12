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
        test1 = {
            "name": "Lift",
            "joint": robot.lift,
            "move_to": 0.8,
            "move_by": -0.4,
            "velocity": 0.05,
        }
        test2 = {
            "name": "Arm",
            "joint": robot.arm,
            "move_to": 0.4,
            "move_by": -0.2,
            "velocity": 0.05,
        }
        joint = test2["joint"]

        pass  # Breakpoint for debugging

        print(f"{test1['name']} status")
        status = joint.status
        print(
            f"{test1['name']} range: {joint.total_range} m, "
            f"position: {status['pos']} m, velocity: {status['vel']} m/s\n"
        )

        print("Move to position\n")
        joint.move_to(test1["move_to"])
        robot.push_command()  # Push command to robot
        robot.wait_command()

        print("Move by a relative amount\n")
        joint.move_by(test1["move_by"])
        robot.push_command()  # Push command to robot
        robot.wait_command()  # Wait until lift reaches the setpoint

        print("Move at velocity\n")
        joint.set_velocity(test1["velocity"])
        robot.push_command()  # Push command to robot
        time.sleep(2)  # Let the lift move for 2 seconds
        joint.set_velocity(0.0)  # Stop the joint
        robot.push_command()  # Push command to robot

    finally:
        robot.stop()  # Stop the robot and clean up
