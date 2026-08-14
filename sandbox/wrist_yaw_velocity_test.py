import threading
import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


TARGET = 0.5
SAMPLE_PERIOD = 0.1


def main():
    stretch = robot.Robot()
    stretch.startup()

    try:
        wrist = stretch.end_of_arm
        joint = wrist.get_joint("wrist_yaw")

        wrist.move_to("wrist_yaw", 0.0)
        stretch.wait_command()

        finished = threading.Event()

        def wait_for_command():
            stretch.wait_command()
            finished.set()

        started = time.perf_counter()
        wrist.move_to("wrist_yaw", TARGET)
        threading.Thread(target=wait_for_command, daemon=True).start()

        while not finished.is_set():
            status = joint.status
            elapsed = time.perf_counter() - started
            print(
                f"{elapsed:.2f} "
                f"pos={status['pos']:.4f} "
                f"vel={status['vel']:.4f} "
                f"error={TARGET - status['pos']:.4f}"
            )
            time.sleep(SAMPLE_PERIOD)

        status = joint.status
        elapsed = time.perf_counter() - started
        print(
            f"done {elapsed:.2f} "
            f"pos={status['pos']:.4f} "
            f"vel={status['vel']:.4f} "
            f"error={TARGET - status['pos']:.4f}"
        )
    finally:
        stretch.stop()


if __name__ == "__main__":
    main()
