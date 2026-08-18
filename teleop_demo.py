import time

from stretch_tools import IS_STRETCH_ENV, NormVelController, TeleopProvider

if IS_STRETCH_ENV:
    import stretch_body.robot as robot
else:
    import stretch_mujoco_api.robot as robot

PRINT_PERIOD = 0.5


def main():
    stretch = robot.Robot()
    if not stretch.startup():
        return

    stretch.enable_collision_mgmt()
    controller = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)
    last_print = 0.0

    try:
        while True:
            command = {}
            # Send velocity command with manual override
            command = teleop.get_manual_override(command)
            controller.set_command(command)

            now = time.monotonic()
            if now - last_print >= PRINT_PERIOD:
                active = {
                    name: round(value, 2)
                    for name, value in command.items()
                    if abs(value) > 0.01
                }
                status = stretch.get_status()
                positions = {
                    "lift": round(status["lift"]["pos"], 3),
                    "arm": round(status["arm"]["pos"], 3),
                    "head_pan": round(status["head"]["head_pan"]["pos"], 3),
                    "head_tilt": round(status["head"]["head_tilt"]["pos"], 3),
                    **{
                        name: round(joint["pos"], 3)
                        for name, joint in status["end_of_arm"].items()
                    },
                }
                print(active or "idle", positions)
                last_print = now

    except KeyboardInterrupt:
        pass
    finally:
        stretch.stop()

if __name__ == "__main__":
    main()
