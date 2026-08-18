try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


if __name__ == "__main__":
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()
    gripper = stretch.end_of_arm.get_joint("stretch_gripper")

    try:
        stretch.lift.move_to(0.41)
        stretch.push_command()
        stretch.wait_command()

        gripper.move_to(50)
        stretch.push_command()
        stretch.wait_command()

        stretch.arm.move_to(0.14)
        stretch.push_command()
        stretch.wait_command()

        gripper.move_to(0.5)
        stretch.push_command()
        stretch.wait_command()

        stretch.lift.move_to(0.6)
        stretch.push_command()
        stretch.wait_command()
    finally:
        stretch.stop()
