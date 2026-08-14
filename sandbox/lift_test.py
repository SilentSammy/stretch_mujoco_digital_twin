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
        pass # Breakpoint for debugging

        # --- LIFT COMMANDS --- (arm works the same way)
        print("Lift status")
        lift_status = robot.lift.status
        lift_range = robot.lift.total_range
        lift_pos_m = lift_status['pos']
        lift_vel_ms = lift_status['vel']
        print(f"Lift range: {lift_range} m, position: {lift_pos_m} m, velocity: {lift_vel_ms} m/s\n")

        print("Move to position\n")
        robot.lift.move_to(0.8)  # Move lift to specified position (in meters)
        robot.push_command()  # Push command to robot
        robot.wait_command()

        print("Move by a relative amount\n")
        robot.lift.move_by(-0.4)  # Move lift by specified amount (in meters)
        robot.push_command()  # Push command to robot
        robot.wait_command()  # Wait until lift reaches the setpoint

        print("Move at velocity\n")
        robot.lift.set_velocity(0.05)  # Move lift at specified velocity (in meters per second)
        robot.push_command()  # Push command to robot
        time.sleep(2)  # Let the lift move for 2 seconds
        robot.lift.set_velocity(0.0)  # Stop the lift
        robot.push_command()  # Push command to robot
    finally:
        robot.stop()  # Stop the robot and clean up
