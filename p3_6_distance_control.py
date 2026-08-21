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


KP = 0.5
KP_DISTANCE = 10.0
TARGET_DISTANCE = 0.75


def detect_object(frame):
    min_area = 100
    lower_1 = np.array([0, 100, 100])
    upper_1 = np.array([10, 255, 255])
    lower_2 = np.array([170, 100, 100])
    upper_2 = np.array([179, 255, 255])

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_1, upper_1)
    mask |= cv2.inRange(hsv, lower_2, upper_2)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if cv2.contourArea(contour) <= min_area or not moments["m00"]:
        return None

    x = int(moments["m10"] / moments["m00"])
    y = int(moments["m01"] / moments["m00"])
    return x, y


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
    head_forward = StateController(
        stretch,
        {"head_pan_counterclockwise": 0.0},
    )
    teleop = TeleopProvider(robot=stretch)
    cameras = Cameras(head_info=HEAD_CAMERA)

    try:
        while True:
            command = scanning_position.get_command()
            command["base_counterclockwise"] = -0.3
            command["base_forward"] = 0.0
            success, frame, depth_frame = cameras.read_head()
            if success:
                center = detect_object(frame)
                if center:
                    x, y = center
                    height, width = frame.shape[:2]
                    horizontal_error = (x - width / 2) / (width / 2)
                    vertical_error = (y - height / 2) / (height / 2)
                    alignment = 1.0 - abs(horizontal_error)
                    distance = HEAD_CAMERA.get_depth(center, depth_frame)

                    command = head_forward.get_command()
                    command["base_counterclockwise"] = -KP * horizontal_error
                    command["head_tilt_up"] = -KP * vertical_error
                    command["base_forward"] = 0.0
                    if distance is not None:
                        distance_error = distance - TARGET_DISTANCE
                        command["base_forward"] = alignment * np.clip(
                            KP_DISTANCE * distance_error, -1.0, 1.0
                        )
                        print(f"Distance: {distance:.3f} m")

                    cv2.circle(frame, center, 5, (0, 0, 255), -1)
                cv2.imshow("Head RGB", frame)
                cv2.imshow("Head Depth", depth_frame)

            command = teleop.get_manual_override(command)
            controller.set_command(command)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        controller.set_command({"base_forward": 0.0, "base_counterclockwise": 0.0})
        cameras.close()
        cv2.destroyAllWindows()
        stretch.stop()


if __name__ == "__main__":
    main()
