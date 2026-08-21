import time

from stretch_tools import NormVelController, TeleopProvider

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


def main():
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()

    controller = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)
    next_pose = time.monotonic() + 5
    next_print = time.monotonic()

    try:
        while True:
            controller.set_command(teleop.get_manual_override({}))

            now = time.monotonic()
            if now >= next_pose:
                stretch.head.pose("ahead")
                stretch.push_command()
                print("Commanded ahead")
                next_pose += 5

            if now >= next_print:
                head = stretch.get_status()["head"]
                print(
                    "pan",
                    round(head["head_pan"]["pos"], 3),
                    "tilt",
                    round(head["head_tilt"]["pos"], 3),
                )
                next_print += 1

            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        stretch.stop()


if __name__ == "__main__":
    main()
