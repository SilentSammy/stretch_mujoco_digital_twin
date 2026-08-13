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
        test = test2
        joint = test["joint"]

        def print_status(label):
            status = joint.status
            print(
                f"{label}: position={status['pos']:.2f} m, "
                f"velocity={status['vel']:.2f} m/s"
            )

        pass  # Breakpoint for debugging

        print(f"Testing {test['name']} (range: {joint.total_range} m)\n")
        print_status("Initial state")

        print(f"\nMoving to {test['move_to']} m")
        joint.move_to(test["move_to"])
        robot.push_command()  # Push command to robot
        robot.wait_command()
        print_status("After move_to")

        print(f"\nMoving by {test['move_by']} m")
        joint.move_by(test["move_by"])
        robot.push_command()  # Push command to robot
        robot.wait_command()  # Wait until lift reaches the setpoint
        print_status("After move_by")

        print(f"\nMoving at {test['velocity']} m/s for 2 seconds")
        joint.set_velocity(test["velocity"])
        robot.push_command()  # Push command to robot
        time.sleep(2)  # Let the lift move for 2 seconds
        print_status("After velocity movement")

        joint.set_velocity(0.0)  # Stop the joint
        robot.push_command()  # Push command to robot
        robot.wait_command()
        print_status("After stop")

    finally:
        robot.stop()  # Stop the robot and clean up
