import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


DURATION = 0.75
SETTLE_TIME = 0.5
SAMPLE_PERIOD = 0.1


def compact_status(base):
    status = base.status
    left = status.get("left_wheel", {})
    right = status.get("right_wheel", {})
    return {
        "x": status.get("x"),
        "y": status.get("y"),
        "theta": status.get("theta"),
        "x_vel": status.get("x_vel"),
        "theta_vel": status.get("theta_vel"),
        "left": {
            "pos": left.get("pos"),
            "vel": left.get("vel"),
            "near_vel_setpoint": left.get("near_vel_setpoint"),
            "is_moving": left.get("is_moving"),
            "is_mg_moving": left.get("is_mg_moving"),
        },
        "right": {
            "pos": right.get("pos"),
            "vel": right.get("vel"),
            "near_vel_setpoint": right.get("near_vel_setpoint"),
            "is_moving": right.get("is_moving"),
            "is_mg_moving": right.get("is_mg_moving"),
        },
    }


def sample_for(base, label, duration):
    started = time.perf_counter()
    while time.perf_counter() - started < duration:
        print(label, f"{time.perf_counter() - started:.2f}", compact_status(base))
        time.sleep(SAMPLE_PERIOD)
    print(label, "end", compact_status(base))


def run_velocity(stretch, label, linear, angular):
    stretch.base.set_velocity(linear, angular)
    stretch.push_command()
    sample_for(stretch.base, label, DURATION)

    stretch.base.set_velocity(0.0, 0.0)
    stretch.push_command()
    sample_for(stretch.base, f"{label} stop", SETTLE_TIME)


def main():
    stretch = robot.Robot()
    stretch.startup()

    try:
        print("initial", compact_status(stretch.base))

        run_velocity(stretch, "translate +", 0.05, 0.0)
        run_velocity(stretch, "translate -", -0.05, 0.0)

        run_velocity(stretch, "rotate +", 0.0, 0.20)
        run_velocity(stretch, "rotate -", 0.0, -0.20)

        run_velocity(stretch, "arc +", 0.04, 0.15)
        run_velocity(stretch, "arc -", -0.04, -0.15)
    finally:
        stretch.base.set_velocity(0.0, 0.0)
        stretch.push_command()
        stretch.stop()


if __name__ == "__main__":
    main()
