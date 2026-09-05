import math
import time

import cv2
import numpy as np

from stretch_tools import (
    Cameras,
    HEAD_CAMERA,
    NormVelController,
    StateController,
    TeleopProvider,
)

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
    cameras = Cameras(head_info=HEAD_CAMERA)
    lower_1 = np.array([0, 100, 100])
    upper_1 = np.array([10, 255, 255])
    lower_2 = np.array([170, 100, 100])
    upper_2 = np.array([179, 255, 255])

    try:
        while True:
            command = scanning_position.get_command()
            command["base_counterclockwise"] = -0.3
            success, frame, depth_frame = cameras.read_head()
            if success:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_1, upper_1)
                mask |= cv2.inRange(hsv, lower_2, upper_2)
                contours, _ = cv2.findContours(
                    mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )
                if contours:
                    contour = max(contours, key=cv2.contourArea)
                    moments = cv2.moments(contour)
                    if cv2.contourArea(contour) > 100 and moments["m00"]:
                        x = int(moments["m10"] / moments["m00"])
                        y = int(moments["m01"] / moments["m00"])
                        print(x, y)
                cv2.imshow("Head RGB", frame)
                cv2.imshow("Head Depth", depth_frame)
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
