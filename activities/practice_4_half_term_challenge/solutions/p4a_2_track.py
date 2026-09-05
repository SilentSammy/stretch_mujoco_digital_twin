import math
import time

import cv2
import numpy as np

from stretch_tools import HEAD_CAMERA, WRIST_CAMERA, NormVelController, RobotTransforms, StateController, TeleopProvider, close_cameras

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


SCAN = 1
TRACK = 2
NAVIGATE = 3
DEPLOY = 4
CENTER = 5
REACH = 6
GRAB = 7

KP_HEAD = 0.5
KP_WRIST = 0.5
KP_ARM = 5.0
KP_ROTATION = 1.0
KP_DISTANCE = 5.0
TARGET_DISTANCE = 0.6
TARGET_WRIST_DISTANCE = 0.15
NAVIGATION_MAX_DISTANCE = 3.0
WRIST_CENTER_TOLERANCE = 0.75
ARM_DISTANCE_TOLERANCE = 0.025
GRAB_POSITION_SETTLE_TIME = 0.5
LIFT_TOP = 1.1
FLANK_ANGLE = -math.pi / 2
FLANK_TOLERANCE = math.radians(5)
WRIST_X_OFFSET = 25
WRIST_Y_OFFSET = 75
SCAN_SPEED = -0.2


def detect_object(frame, depth_frame=None, max_distance=None, camera_info=None):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 100, 100]), np.array([10, 255, 255]))
    mask |= cv2.inRange(hsv, np.array([170, 100, 100]), np.array([179, 255, 255]))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


def run(stage):
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
    deploy_wrist = StateController(
        stretch,
        {
            "wrist_pitch_up": 0.0,
            "wrist_roll_counterclockwise": 0.0,
            "wrist_yaw_counterclockwise": 0.0,
        },
    )
    gripper_open_controller = StateController(stretch, {"gripper_open": 130.0})
    gripper_close_controller = StateController(stretch, {"gripper_open": -25.0})

    command = None
    head_rgb = head_depth = head_center = object_T = None
    wrist_rgb = wrist_depth = wrist_center = None
    within_distance = False
    gripper_ready = False
    gripper_closing = False
    grab_position_started = None
    ready_to_close = False
    grabbed = False
    phase = "navigate"

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
        nonlocal within_distance, phase
        object_x, object_y, object_z = object_T[:3, 3]
        distance = math.hypot(object_x, object_y)
        facing_error = math.atan2(object_y, object_x)
        within_distance = (
            0.5 <= distance <= 0.7
            if within_distance
            else 0.55 <= distance <= 0.65
        )

        if within_distance:
            flank_error = math.atan2(
                math.sin(facing_error - FLANK_ANGLE),
                math.cos(facing_error - FLANK_ANGLE),
            )
            command["base_counterclockwise"] = KP_ROTATION * flank_error
            if abs(flank_error) <= FLANK_TOLERANCE:
                lift_controller.desired_state["lift_up"] = object_z
                if stage >= DEPLOY and lift_controller.is_at_goal():
                    phase = "grab"
                    print(f"\nPhase: {phase}")
        else:
            alignment = max(0.0, 1.0 - abs(facing_error) / (math.pi / 2))
            command["base_counterclockwise"] = KP_ROTATION * facing_error
            command["base_forward"] = alignment * KP_DISTANCE * (
                distance - TARGET_DISTANCE
            )

    def center_object_with_wrist():
        x, y = wrist_center
        height, width = wrist_rgb.shape[:2]
        horizontal_error = (x - width / 2 - WRIST_X_OFFSET) / (width / 2)
        vertical_error = (y - height / 2 - WRIST_Y_OFFSET) / (height / 2)
        command["wrist_yaw_counterclockwise"] = -KP_WRIST * horizontal_error
        command["wrist_pitch_up"] = -KP_WRIST * vertical_error
        cv2.circle(wrist_rgb, wrist_center, 5, (0, 255, 0), -1)
        return horizontal_error, vertical_error

    def reach_for_object(horizontal_error, vertical_error):
        nonlocal ready_to_close
        if not gripper_ready:
            return
        distance = WRIST_CAMERA.get_depth(wrist_center, wrist_depth, sample_radius=15)
        if distance is None:
            return
        center_error = math.hypot(horizontal_error, vertical_error)
        authority = max(0.0, 1.0 - center_error)
        command["arm_out"] = authority * KP_ARM * (
            distance - TARGET_WRIST_DISTANCE
        )
        ready_to_close = (
            center_error <= WRIST_CENTER_TOLERANCE
            and abs(distance - TARGET_WRIST_DISTANCE) <= ARM_DISTANCE_TOLERANCE
        )

    def update_gripper():
        nonlocal grab_position_started, gripper_closing, grabbed
        if not gripper_closing:
            if ready_to_close:
                if grab_position_started is None:
                    grab_position_started = time.monotonic()
                elif time.monotonic() - grab_position_started >= GRAB_POSITION_SETTLE_TIME:
                    gripper_closing = True
                    print("\nGrabbing position reached")
            else:
                grab_position_started = None

        command.update(
            gripper_close_controller.get_command()
            if gripper_closing
            else gripper_open_controller.get_command()
        )
        if gripper_closing and not grabbed and gripper_close_controller.is_at_goal():
            grabbed = True
            lift_controller.desired_state["lift_up"] = LIFT_TOP
            print("\nObject grabbed")
        if grabbed:
            command.update(
                {
                    "arm_out": 0.0,
                    "wrist_pitch_up": 0.0,
                    "wrist_roll_counterclockwise": 0.0,
                    "wrist_yaw_counterclockwise": 0.0,
                }
            )

    try:
        while True:
            command = {}
            if phase == "navigate":
                command.update(arm_stow.get_command())
                success, head_rgb, head_depth = HEAD_CAMERA.get_frames()
                if not success:
                    return
                head_center = detect_object(
                    head_rgb, head_depth, NAVIGATION_MAX_DISTANCE, HEAD_CAMERA
                )
                stowed = arm_stow.is_at_goal()
                if head_center is None:
                    if stowed and stage >= SCAN:
                        scan()
                else:
                    if stage >= TRACK:
                        track_with_head()
                    if stage >= NAVIGATE:
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
            else:
                ready_to_close = False
                command["arm_out"] = 0.0
                command.update(deploy_wrist.get_command())
                if gripper_open_controller.is_at_goal():
                    gripper_ready = True

                success, wrist_rgb, wrist_depth = WRIST_CAMERA.get_frames()
                if success:
                    wrist_center = detect_object(wrist_rgb)
                    if wrist_center and not grabbed and stage >= CENTER:
                        errors = center_object_with_wrist()
                        if stage >= REACH:
                            reach_for_object(*errors)
                    cv2.imshow("Wrist RGB", wrist_rgb)
                    depth_display = cv2.normalize(
                        wrist_depth, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                    )
                    cv2.imshow("Wrist Depth", depth_display)

                if stage >= GRAB:
                    update_gripper()
                else:
                    command.update(gripper_open_controller.get_command())

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
    run(TRACK)
