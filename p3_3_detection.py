import math
import time

import cv2
import numpy as np

from stretch_tools import Cameras, NormVelController, StateController, TeleopProvider
from stretch_tools.cameras import HEAD_COLOR

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


# LOWER_BLUE = np.array([110, 100, 100])
# UPPER_BLUE = np.array([130, 255, 255])
LOWER_RED_1 = np.array([0, 100, 100])
UPPER_RED_1 = np.array([10, 255, 255])
LOWER_RED_2 = np.array([170, 100, 100])
UPPER_RED_2 = np.array([179, 255, 255])
MIN_AREA = 100


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
            success, frame = cameras.read(HEAD_COLOR)
            if success:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, LOWER_RED_1, UPPER_RED_1)
                mask |= cv2.inRange(hsv, LOWER_RED_2, UPPER_RED_2)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > MIN_AREA:
                    command = {"base_counterclockwise": 0.0}
                cv2.imshow("Head RGB", frame)
                cv2.imshow("Red Mask", mask)

            command = teleop.get_manual_override(command)
            controller.set_command(command)

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
