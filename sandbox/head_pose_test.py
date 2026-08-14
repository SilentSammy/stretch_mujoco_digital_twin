import threading
import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


def main():
    stretch = robot.Robot()
    stretch.startup()

    try:
        for pose in ("ahead", "tool", "wheels", "back"):
            print(f"\nStarting {pose}")
            stretch.head.pose(pose)
            waiter = threading.Thread(target=stretch.wait_command, daemon=True)
            waiter.start()
            started = time.perf_counter()

            while waiter.is_alive() and time.perf_counter() - started < 10:
                diagnostics = {}
                for name in stretch.head.joints:
                    joint = stretch.head.get_joint(name)
                    status = joint.status
                    target = getattr(joint, "_motion", (None,))[0]
                    diagnostics[name] = {
                        "target": target,
                        "pos": round(status["pos"], 4),
                        "error": None if target is None else round(target - status["pos"], 4),
                        "vel": round(status["vel"], 4),
                        "waiting": joint in getattr(stretch, "_waiting_for", set()),
                    }
                print(f"{time.perf_counter() - started:.1f}s", diagnostics)
                time.sleep(0.1)

            if waiter.is_alive():
                print(f"TIMEOUT waiting for {pose}")
                break

            pan = stretch.head.get_joint("head_pan").status["pos"]
            tilt = stretch.head.get_joint("head_tilt").status["pos"]
            print(pose, {"head_pan": pan, "head_tilt": tilt})

    finally:
        stretch.stop()


if __name__ == "__main__":
    main()
