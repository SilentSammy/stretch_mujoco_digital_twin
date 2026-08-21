import math
import time

import cv2

from stretch_tools import Cameras, NormVelController, StateController, TeleopProvider
from stretch_tools.cameras import HEAD_COLOR

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
    cameras = Cameras()

    try:
        while True:
            command = scanning_position.get_command()
            command["base_counterclockwise"] = -0.5
            command = teleop.get_manual_override(command)
            controller.set_command(command)

            success, frame = cameras.read(HEAD_COLOR)
            if success:
                cv2.imshow("Head RGB", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        controller.set_command({"base_counterclockwise": 0.0})
        cameras.close()
        cv2.destroyAllWindows()
        stretch.stop()


if __name__ == "__main__":
    main()
