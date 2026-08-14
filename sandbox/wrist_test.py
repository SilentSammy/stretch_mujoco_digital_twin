import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


def compact_status(joint):
    status = joint.status
    return {
        "keys": list(status),
        "pos": status.get("pos"),
        "vel": status.get("vel"),
        "stalled": status.get("stalled"),
    }


def main():
    stretch = robot.Robot()
    stretch.startup()

    try:
        wrist = stretch.end_of_arm
        names = [name for name in wrist.joints if name.startswith("wrist_")]
        print("joints", wrist.joints, type(wrist.joints).__name__)

        for name in names:
            print(name, "initial", compact_status(wrist.get_joint(name)))

        tests = {
            "wrist_yaw": (0.5, -0.5),
            "wrist_pitch": (-0.5, 0.5),
            "wrist_roll": (1.0, -1.0),
        }

        for name in names:
            move_to, move_by = tests[name]
            joint = wrist.get_joint(name)

            started = time.perf_counter()
            wrist.move_to(name, move_to)
            time.sleep(0.2)
            print(name, "moving", compact_status(joint))
            stretch.wait_command()
            print(
                name,
                "after move_to",
                round(time.perf_counter() - started, 3),
                compact_status(joint),
            )

            started = time.perf_counter()
            wrist.move_by(name, move_by)
            time.sleep(0.2)
            print(name, "moving_by", compact_status(joint))
            stretch.wait_command()
            print(
                name,
                "after move_by",
                round(time.perf_counter() - started, 3),
                compact_status(joint),
            )

    finally:
        stretch.stop()


if __name__ == "__main__":
    main()
