import math

import cv2
import numpy as np

from stretch_tools import HEAD_CAMERA, NormVelController, RobotTransforms, StateController, TeleopProvider, close_cameras

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


KP_HEAD = 0.5
KP_ROTATION = 1.0
KP_DISTANCE = 5.0
TARGET_DISTANCE = 0.6
NAVIGATION_MAX_DISTANCE = 3.0
FLANK_ANGLE = -math.pi / 2
FLANK_TOLERANCE = math.radians(5)
LIFT_HEIGHT_OFFSET = -0.0
SCAN_SPEED = -0.2


def detect_object(frame, depth_frame=None, max_distance=None, camera_info=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask |= cv2.inRange(hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in sorted(contours, key=cv2.contourArea, reverse=True):
        if cv2.contourArea(contour) <= 100:
            break
        moments = cv2.moments(contour)
        if not moments["m00"]:
            continue

        center = (
            int(moments["m10"] / moments["m00"]),
            int(moments["m01"] / moments["m00"]),
        )
        if depth_frame is not None and max_distance is not None:
            distance = camera_info.get_depth(center, depth_frame)
            if distance is None or distance > max_distance:
                continue
        return center

    return None


def locate_object(center, depth_frame, camera_info, camera_T):
    depth = camera_info.get_depth(center, depth_frame)
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
    transforms = RobotTransforms(stretch)

    arm_stow = StateController(
        stretch,
        {
            "arm_out": 0.0,
            "wrist_pitch_up": -0.625,
            "wrist_roll_counterclockwise": 0.0,
            "wrist_yaw_counterclockwise": 2.994,
            "gripper_open": 1.0,
        },
    )
    lift_controller = StateController(stretch, {"lift_up": 0.3})
    scanning_position = StateController(
        stretch,
        {
            "head_pan_counterclockwise": 0.0,
            "head_tilt_up": math.radians(-30),
        },
    )

    command = None
    head_rgb, head_depth = None, None
    head_center = None
    object_T = None

    within_distance = False
    stowed = False

    def scan():
        command.update(scanning_position.get_command())
        command["base_counterclockwise"] = SCAN_SPEED

    def track_with_head():
        x, y = head_center
        height, width = head_rgb.shape[:2]
        horizontal_error = (x - width / 2) / (width / 2)
        vertical_error = (y - height / 2) / (height / 2)
        command["head_pan_counterclockwise"] = -KP_HEAD * horizontal_error
        command["head_tilt_up"] = -KP_HEAD * vertical_error

    def navigate_to_object():
        nonlocal within_distance

        object_x, object_y, object_z = object_T[:3, 3]
        distance = math.hypot(object_x, object_y)
        facing_error = math.atan2(object_y, object_x)

        if within_distance:
            within_distance = 0.5 <= distance <= 0.7
        else:
            within_distance = 0.55 <= distance <= 0.65

        if within_distance:
            flank_error = math.atan2(
                math.sin(facing_error - FLANK_ANGLE),
                math.cos(facing_error - FLANK_ANGLE),
            )
            command["base_counterclockwise"] = KP_ROTATION * flank_error
            if abs(flank_error) <= FLANK_TOLERANCE:
                lift_controller.desired_state["lift_up"] = (
                    object_z - LIFT_HEIGHT_OFFSET
                )
        else:
            alignment = max(0.0, 1.0 - abs(facing_error) / (math.pi / 2))
            command["base_counterclockwise"] = KP_ROTATION * facing_error
            command["base_forward"] = alignment * KP_DISTANCE * (
                distance - TARGET_DISTANCE
            )

    try:
        while True:
            command = {}
            command.update(arm_stow.get_command())
            success, head_rgb, head_depth = HEAD_CAMERA.get_frames()
            if not success:
                return

            head_center = detect_object(
                head_rgb, head_depth, NAVIGATION_MAX_DISTANCE, HEAD_CAMERA
            )
            stowed = arm_stow.is_at_goal()

            if head_center is None:
                if stowed:
                    scan()
            else:
                track_with_head()
                object_T = locate_object(
                    head_center,
                    head_depth,
                    HEAD_CAMERA,
                    transforms.get_cam_T(HEAD_CAMERA),
                )
                if object_T is not None and stowed:
                    navigate_to_object()
                cv2.circle(head_rgb, head_center, 5, (0, 255, 0), -1)

            cv2.imshow("Head RGB", head_rgb)

            command.update(lift_controller.get_command())
            command = teleop.get_manual_override(command)
            controller.set_command(command)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except KeyboardInterrupt:
        pass
    finally:
        close_cameras()
        cv2.destroyAllWindows()
        stretch.stop()


if __name__ == "__main__":
    main()
