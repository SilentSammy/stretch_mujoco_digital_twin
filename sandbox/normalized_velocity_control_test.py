import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot

from stretch_tools.normalized_velocity_control import NormalizedVelocityControl


ACTIVE_TIME = 3.0
STOP_TIME = 0.75
SAMPLE_PERIOD = 0.25
NORMALIZED_VELOCITY = 0.5


def zero_command():
    return {name: 0.0 for name in NormalizedVelocityControl.MAX_VELOCITIES}


def get_status(stretch, name):
    if name == "base_counterclockwise":
        status = stretch.base.status
        return {
            "pos": status["theta"],
            "vel": status["theta_vel"],
        }
    if name == "lift_up":
        return stretch.lift.status
    if name == "arm_out":
        return stretch.arm.status
    if name.startswith("head_"):
        joint_name = name.removesuffix("_counterclockwise").removesuffix("_up")
        return stretch.head.get_joint(joint_name).status

    joint_name = name.removesuffix("_counterclockwise").removesuffix("_up")
    if name == "gripper_open":
        joint_name = "stretch_gripper"
    return stretch.end_of_arm.get_joint(joint_name).status


def print_status(stretch, name, label, elapsed):
    status = get_status(stretch, name)
    output = [
        name,
        label,
        f"{elapsed:.2f}",
        f"pos={status['pos']:.4f}",
        f"vel={status['vel']:.4f}",
    ]
    if "pos_pct" in status:
        output.append(f"pos_pct={status['pos_pct']:.2f}")
    print(*output)


def sample(control, stretch, name, label, duration, command):
    started = time.perf_counter()
    while time.perf_counter() - started < duration:
        control.set_command(command)
        elapsed = time.perf_counter() - started
        print_status(stretch, name, label, elapsed)
        time.sleep(SAMPLE_PERIOD)
    control.set_command(command)
    print_status(stretch, name, label, time.perf_counter() - started)


def main():
    stretch = robot.Robot()
    if not stretch.startup():
        return

    stretch.enable_collision_mgmt()
    control = NormalizedVelocityControl(stretch)
    names = [
        name
        for name in NormalizedVelocityControl.MAX_VELOCITIES
        if name != "base_forward"
    ]

    try:
        for name in names:
            for direction in (NORMALIZED_VELOCITY, -NORMALIZED_VELOCITY):
                command = zero_command()
                command[name] = direction
                sample(
                    control,
                    stretch,
                    name,
                    f"command={direction:+.1f}",
                    ACTIVE_TIME,
                    command,
                )

                stopped = zero_command()
                sample(
                    control,
                    stretch,
                    name,
                    "stopped",
                    STOP_TIME,
                    stopped,
                )
    finally:
        control.set_command(zero_command())
        stretch.stop()


if __name__ == "__main__":
    main()
