import threading
import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


TRANSLATION = 0.10
ROTATION = 0.20
SAMPLE_PERIOD = 0.1
COMMAND_TIMEOUT = 30.0


def compact_status(base):
    status = base.status
    return {
        "x": status.get("x"),
        "y": status.get("y"),
        "theta": status.get("theta"),
        "x_vel": status.get("x_vel"),
        "y_vel": status.get("y_vel"),
        "theta_vel": status.get("theta_vel"),
    }


def sample_motion(stretch, label):
    finished = threading.Event()

    def wait_for_command():
        stretch.wait_command()
        finished.set()

    started = time.perf_counter()
    threading.Thread(target=wait_for_command, daemon=True).start()

    while not finished.is_set():
        elapsed = time.perf_counter() - started
        print(label, f"{elapsed:.2f}", compact_status(stretch.base))
        if elapsed >= COMMAND_TIMEOUT:
            print(label, "timeout")
            return
        time.sleep(SAMPLE_PERIOD)

    print(label, "done", f"{time.perf_counter() - started:.2f}", compact_status(stretch.base))


def run_translation(stretch, distance):
    stretch.base.translate_by(distance)
    stretch.push_command()
    sample_motion(stretch, f"translate {distance:+.2f}")


def run_rotation(stretch, angle):
    stretch.base.rotate_by(angle)
    stretch.push_command()
    sample_motion(stretch, f"rotate {angle:+.2f}")


def main():
    stretch = robot.Robot()
    stretch.startup()

    try:
        print("base keys", list(stretch.base.status))
        print("left wheel keys", list(stretch.base.status.get("left_wheel", {})))
        print("right wheel keys", list(stretch.base.status.get("right_wheel", {})))
        print("initial", compact_status(stretch.base))

        run_translation(stretch, TRANSLATION)
        run_translation(stretch, -TRANSLATION)
        run_rotation(stretch, ROTATION)
        run_rotation(stretch, -ROTATION)
    finally:
        stretch.base.set_velocity(0.0, 0.0)
        stretch.push_command()
        stretch.stop()


if __name__ == "__main__":
    main()
