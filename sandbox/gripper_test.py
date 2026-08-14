import threading
import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


DISTANCE = 50.0
SAMPLE_PERIOD = 0.1
COMMAND_TIMEOUT = 5.0


def compact_status(joint):
    status = joint.status
    return {
        "pos": status.get("pos"),
        "vel": status.get("vel"),
        "effort": status.get("effort"),
        "stalled": status.get("stalled"),
    }


def sample_motion(stretch, joint, label):
    finished = threading.Event()

    def wait_for_command():
        stretch.wait_command()
        finished.set()

    started = time.perf_counter()
    threading.Thread(target=wait_for_command, daemon=True).start()

    while not finished.is_set():
        elapsed = time.perf_counter() - started
        print(label, f"{elapsed:.2f}", compact_status(joint))
        if elapsed >= COMMAND_TIMEOUT:
            print(label, "timeout")
            return
        time.sleep(SAMPLE_PERIOD)

    print(label, "done", f"{time.perf_counter() - started:.2f}", compact_status(joint))


def main():
    stretch = robot.Robot()
    stretch.startup()

    try:
        gripper = stretch.end_of_arm.get_joint("stretch_gripper")
        print("joints", stretch.end_of_arm.joints, type(stretch.end_of_arm.joints).__name__)
        print("status keys", list(gripper.status))
        print("initial", compact_status(gripper))

        stretch.end_of_arm.move_by("stretch_gripper", DISTANCE)
        sample_motion(stretch, gripper, f"+{DISTANCE:g}")

        stretch.end_of_arm.move_by("stretch_gripper", -DISTANCE)
        sample_motion(stretch, gripper, f"-{DISTANCE:g}")
    finally:
        stretch.stop()


if __name__ == "__main__":
    main()
