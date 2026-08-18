import time
import math
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

    try:
        while True:
            alg_command = {
                'lift_up': 0.5 * math.sin(0.5 * time.time())
            }

            command = teleop.get_manual_override(alg_command)
            controller.set_command(command)
            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        controller.set_command(controller.zero_cmd)
        stretch.stop()

if __name__ == "__main__":
    main()
