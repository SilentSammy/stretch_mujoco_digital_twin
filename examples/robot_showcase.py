import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot

if __name__ == "__main__":
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()
    try:
        base = stretch.base
        lift = stretch.lift
        arm = stretch.arm
        head_pan = stretch.head.get_joint("head_pan")
        head_tilt = stretch.head.get_joint("head_tilt")
        wrist_pitch = stretch.end_of_arm.get_joint("wrist_pitch")
        wrist_roll = stretch.end_of_arm.get_joint("wrist_roll")
        wrist_yaw = stretch.end_of_arm.get_joint("wrist_yaw")
        gripper = stretch.end_of_arm.get_joint("stretch_gripper")

        print("Homed:", stretch.is_homed())
        if not stretch.is_homed():
            stretch.home()
            stretch.push_command()

        print("Stowing")
        stretch.stow()
        stretch.push_command()

        print("Robot status:", stretch.get_status())
        print("Lift range:", lift.total_range)

        print("Moving lift to 0.5 m")
        lift.move_to(0.5, v_m=0.08)
        stretch.push_command()
        stretch.wait_command()

        print("Extending arm by 0.1 m")
        arm.move_by(0.1, v_m=0.08)
        stretch.push_command()
        stretch.wait_command()

        print("Moving head to the tool pose")
        stretch.head.pose("tool")
        stretch.push_command()
        stretch.wait_command()

        print("Moving wrist pitch to -0.4 rad")
        wrist_pitch.move_to(-0.4)
        stretch.push_command()
        stretch.wait_command()

        print("Rotating wrist roll by 0.35 rad")
        wrist_roll.move_by(0.35)
        stretch.push_command()
        stretch.wait_command()

        print("Moving wrist yaw at -0.3 rad/s")
        wrist_yaw.set_velocity(-0.3)
        stretch.push_command()
        time.sleep(0.5)
        wrist_yaw.set_velocity(0.0)
        stretch.push_command()
        stretch.wait_command()

        print("Opening gripper to 30%")
        gripper.move_to(30.0)
        stretch.push_command()
        stretch.wait_command()

        print("Translating base by 0.1 m")
        base.translate_by(0.1)
        stretch.push_command()
        stretch.wait_command()
        base.translate_by(-0.1)
        stretch.push_command()
        stretch.wait_command()

        print("Rotating base by 0.25 rad")
        base.rotate_by(0.25)
        stretch.push_command()
        stretch.wait_command()
        base.rotate_by(-0.25)
        stretch.push_command()
        stretch.wait_command()

        print("Driving base with linear and angular velocity")
        for _ in range(10):
            base.set_velocity(0.05, 0.1)
            stretch.push_command()
            time.sleep(0.1)
        base.set_velocity(0.0, 0.0)
        stretch.push_command()

        print("Final joint positions:", {
            "lift": lift.status["pos"],
            "arm": arm.status["pos"],
            "head_pan": head_pan.status["pos"],
            "head_tilt": head_tilt.status["pos"],
            "wrist_pitch": wrist_pitch.status["pos"],
            "wrist_roll": wrist_roll.status["pos"],
            "wrist_yaw": wrist_yaw.status["pos"],
            "gripper": gripper.status["pos"],
        })

        print("Returning to stow pose")
        stretch.stow()
        stretch.push_command()
    finally:
        stretch.stop()
