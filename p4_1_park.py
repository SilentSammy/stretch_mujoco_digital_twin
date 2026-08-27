import math
import time

import cv2
import numpy as np

from stretch_tools import (
    Cameras,
    HEAD_CAMERA,
    NormVelController,
    RobotTransforms,
    StateController,
    TeleopProvider,
)

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


KP_HEAD = 0.5
KP_ROTATION = 1.0
KP_DISTANCE = 5.0
TARGET_DISTANCE = 0.5
FLANK_ANGLE = -math.pi / 2
FLANK_TOLERANCE = math.radians(5)
LIFT_HEIGHT_OFFSET = 0.0
SCAN_SPEED = -0.2


def detect_object(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask |= cv2.inRange(hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if cv2.contourArea(contour) <= 100 or not moments["m00"]:
        return None

    return (
        int(moments["m10"] / moments["m00"]),
        int(moments["m01"] / moments["m00"]),
    )


def locate_object(center, depth_frame, camera_info, camera_T):
    depth = camera_info.get_depth(center, depth_frame, sample_radius=15)
    if depth is None:
        return None

    x, y = camera_info.pixel_to_normalized(center)
    object_T = np.eye(4)
    object_T[:3, 3] = [x * depth, y * depth, depth]
    return camera_T @ object_T


def main():
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()

    controller = NormVelController(stretch, safe_base_mode=True)
    teleop = TeleopProvider(robot=stretch)
    cameras = Cameras(head_info=HEAD_CAMERA)
    transforms = RobotTransforms(stretch)
    navigation_stow = StateController(
        stretch,
        {
            "arm_out": 0.0,
            "wrist_pitch_up": -0.625,
            "wrist_roll_counterclockwise": 0.0,
            "wrist_yaw_counterclockwise": 2.994,
            "gripper_open": 1.0,
        },
    )
    lift_controller = StateController(
        stretch,
        {"lift_up": 0.3},
    )
    scanning_position = StateController(
        stretch,
        {
            "head_pan_counterclockwise": 0.0,
            "head_tilt_up": math.radians(-30),
        },
    )

    within_distance = False

    try:
        while True:
            command = navigation_stow.get_command()
            command["base_forward"] = 0.0
            command["base_counterclockwise"] = 0.0

            success, frame, depth_frame = cameras.read_head()
            if success:
                center = detect_object(frame)
                stowed = navigation_stow.is_at_goal()

                if center is None:
                    if stowed:
                        command.update(scanning_position.get_command())
                        command["base_counterclockwise"] = SCAN_SPEED
                else:
                    x, y = center
                    height, width = frame.shape[:2]
                    horizontal_error = (x - width / 2) / (width / 2)
                    vertical_error = (y - height / 2) / (height / 2)
                    command["head_pan_counterclockwise"] = (
                        -KP_HEAD * horizontal_error
                    )
                    command["head_tilt_up"] = -KP_HEAD * vertical_error

                    object_T = locate_object(
                        center,
                        depth_frame,
                        cameras.head_info,
                        transforms.get_cam_T(cameras.head_info),
                    )
                    if object_T is not None and stowed:
                        object_x, object_y, object_z = object_T[:3, 3]
                        distance = math.hypot(object_x, object_y)
                        facing_error = math.atan2(object_y, object_x)

                        if within_distance:
                            within_distance = 0.4 <= distance <= 0.6
                        else:
                            within_distance = 0.45 <= distance <= 0.55

                        if within_distance:
                            flank_error = math.atan2(
                                math.sin(facing_error - FLANK_ANGLE),
                                math.cos(facing_error - FLANK_ANGLE),
                            )
                            command["base_counterclockwise"] = np.clip(
                                KP_ROTATION * flank_error,
                                -1.0,
                                1.0,
                            )
                            if abs(flank_error) <= FLANK_TOLERANCE:
                                lift_controller.desired_state["lift_up"] = (
                                    object_z - LIFT_HEIGHT_OFFSET
                                )
                        else:
                            alignment = max(
                                0.0,
                                1.0 - abs(facing_error) / (math.pi / 2),
                            )
                            command["base_counterclockwise"] = np.clip(
                                KP_ROTATION * facing_error,
                                -1.0,
                                1.0,
                            )
                            command["base_forward"] = alignment * np.clip(
                                KP_DISTANCE * (distance - TARGET_DISTANCE),
                                -1.0,
                                1.0,
                            )

                    cv2.circle(frame, center, 5, (0, 255, 0), -1)

                cv2.imshow("Head RGB", frame)

            command.update(lift_controller.get_command())
            command = teleop.get_manual_override(command)
            controller.set_command(command)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        controller.set_command(
            {
                "base_forward": 0.0,
                "base_counterclockwise": 0.0,
                "head_pan_counterclockwise": 0.0,
                "head_tilt_up": 0.0,
            }
        )
        cameras.close()
        cv2.destroyAllWindows()
        stretch.stop()


if __name__ == "__main__":
    main()
