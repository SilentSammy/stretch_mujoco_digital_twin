import math
import time

from stretch_tools import NormVelController, StateController, TeleopProvider

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


def main():
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()

    controller = NormVelController(stretch)
    scanning_position = StateController(
        stretch,
        {
            "head_pan_counterclockwise": 0.0,
            "head_tilt_up": math.radians(-30),
        },
    )
    teleop = TeleopProvider(robot=stretch)

    try:
        while True:
            command = scanning_position.get_command()
            command["base_counterclockwise"] = -0.5
            command = teleop.get_manual_override(command)
            controller.set_command(command)

            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        controller.set_command({"base_counterclockwise": 0.0})
        stretch.stop()


if __name__ == "__main__":
    main()
