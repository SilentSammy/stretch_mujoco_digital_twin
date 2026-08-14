import time

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot

from stretch_tools.normalized_velocity_control import NormalizedVelocityControl
from stretch_tools.teleop_provider import TeleopProvider


UPDATE_PERIOD = 0.1
PRINT_PERIOD = 0.5


def complete_command(command):
    return {
        name: command.get(name, 0.0)
        for name in NormalizedVelocityControl.MAX_VELOCITIES
    }


def main():
    stretch = robot.Robot()
    if not stretch.startup():
        return

    stretch.enable_collision_mgmt()
    control = NormalizedVelocityControl(stretch)
    teleop = TeleopProvider()
    stopped = complete_command({})
    last_print = 0.0

    print("Teleop active. Press Ctrl+C to stop.")
    print("W/S base, A/D turn, Z/X lift, V/C arm, M/N gripper")
    print("U/O wrist roll, I/K wrist pitch, J/L wrist yaw, H toggles head")

    try:
        while True:
            command = complete_command(teleop.get_normalized_velocities())
            control.set_command(command)

            now = time.monotonic()
            if now - last_print >= PRINT_PERIOD:
                active = {
                    name: round(value, 2)
                    for name, value in command.items()
                    if abs(value) > 0.01
                }
                print(active or "idle")
                last_print = now

            time.sleep(UPDATE_PERIOD)
    except KeyboardInterrupt:
        pass
    finally:
        for _ in range(10):
            control.set_command(stopped)
            time.sleep(UPDATE_PERIOD)
        stretch.stop()


if __name__ == "__main__":
    main()
