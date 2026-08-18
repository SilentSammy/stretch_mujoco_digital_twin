import time

import stretch_mujoco_api.robot as robot


if __name__ == "__main__":
    stretch = robot.Robot()
    stretch.startup()
    gripper = stretch.end_of_arm.get_joint("stretch_gripper")

    try:
        commands = (
            ("lift 0.41", lambda: stretch.lift.move_to(0.41)),
            ("gripper 50", lambda: gripper.move_to(50.0)),
            ("arm 0.14", lambda: stretch.arm.move_to(0.14)),
            ("gripper 0.5", lambda: gripper.move_to(0.5)),
        )
        for name, command in commands:
            command()
            stretch.push_command()
            completed = stretch._command_complete.wait(timeout=15.0)
            print(name, completed, stretch.lift.status["pos"], stretch.arm.status["pos"], gripper.status["pos_pct"])
            assert completed
    finally:
        stretch.stop()
