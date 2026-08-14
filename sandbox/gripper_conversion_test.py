import time

import stretch_body.robot as robot


DISTANCE = 50.0
SETTLE_TIME = 0.5


def print_status(label, gripper):
    status = gripper.status
    print(
        label,
        {
            "pos": status.get("pos"),
            "pos_pct": status.get("pos_pct"),
            "gripper_conversion": status.get("gripper_conversion"),
        },
    )


def main():
    stretch = robot.Robot()
    stretch.startup()

    try:
        gripper = stretch.end_of_arm.get_joint("stretch_gripper")
        print_status("initial", gripper)

        stretch.end_of_arm.move_by("stretch_gripper", DISTANCE)
        stretch.wait_command()
        time.sleep(SETTLE_TIME)
        print_status("opened", gripper)

        stretch.end_of_arm.move_by("stretch_gripper", -DISTANCE)
        stretch.wait_command()
        time.sleep(SETTLE_TIME)
        print_status("returned", gripper)
    finally:
        stretch.stop()


if __name__ == "__main__":
    main()
