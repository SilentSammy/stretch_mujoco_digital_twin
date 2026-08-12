import time
import stretch_mujoco_api.robot

if __name__ == "__main__":
    robot = stretch_mujoco_api.robot.Robot()
    robot.startup()
    robot.enable_collision_mgmt()
    try:
        print(f"Limits: {robot.lift}")
        # Start position
        robot.lift.move_to(1.0)
        robot.push_command()
        robot.wait_command()

        # End position
        robot.lift.move_to(0.5)
        robot.push_command()

        prev_lift_pos = robot.lift.status['pos']
        prev_time = time.time()
        for i in range(200):
            dt = time.time() - prev_time
            prev_time = time.time()

            lift_status = robot.lift.status
            lift_pos = lift_status['pos']
            lift_reported_vel = lift_status['vel']

            lift_computed_vel = (lift_pos - prev_lift_pos) / dt
            prev_lift_pos = lift_pos

            print(f"rv: {lift_reported_vel:.3f}, cv: {lift_computed_vel:.3f}, pos: {lift_pos:.3f}, {robot._sim.pull_status().sim_to_real_time_ratio_msg}")
            time.sleep(0.05)
        robot.wait_command()

    finally:
        robot.stop()  # Stop the robot and clean up
