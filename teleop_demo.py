import time

from stretch_tools import IS_STRETCH_ENV, NormVelController, TeleopProvider

if IS_STRETCH_ENV:
    import stretch_body.robot as robot
else:
    import stretch_mujoco_api.robot as robot


UPDATE_PERIOD = 0.1
PRINT_PERIOD = 0.5


def main():
    stretch = robot.Robot()
    if not stretch.startup():
        return

    stretch.enable_collision_mgmt()
    control = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)
    algorithmic_command = {}
    stopped = {name: 0.0 for name in NormVelController.MAX_VELOCITIES}
    last_print = 0.0

    print("Teleop active. Press Ctrl+C to stop.")
    print("W/S base, A/D turn, Z/X lift, V/C arm, M/N gripper")
    print("U/O wrist roll, I/K wrist pitch, J/L wrist yaw, H toggles head")

    try:
        while True:
            command = teleop.get_manual_override(algorithmic_command)
            control.set_command(command)

            now = time.monotonic()
            if now - last_print >= PRINT_PERIOD:
                active = {
                    name: round(value, 2)
                    for name, value in command.items()
                    if abs(value) > 0.01
                }
                output = [active or "idle"]
                if not IS_STRETCH_ENV:
                    output.append(stretch._sim.pull_status().sim_to_real_time_ratio_msg)
                print(*output)
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
