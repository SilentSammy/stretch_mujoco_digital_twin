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
    WRIST_CAMERA,
)

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


KP_BASE_ALIGN = 0.5
KP_TILT = 0.5
KP_ANGLE = 5.0 / math.pi
KP_FORWARD = 2.0
KP_LIFT = 5.0
KP_YAW = 0.5
KP_PITCH = 0.5
KP_ARM = 10.0


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


def add_pose(command, pose):
    command.update(
        {name: velocity for name, velocity in pose.get_command().items() if velocity}
    )


def main():
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()

    controller = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)
    cameras = Cameras()
    transforms = RobotTransforms(stretch)

    stow_pose = StateController(
        stretch,
        {
            "wrist_roll_counterclockwise": 0.0,
            "wrist_yaw_counterclockwise": 0.0,
            "wrist_pitch_up": 0.0,
            "gripper_open": 0.3,
            "arm_out": 0.0,
        },
    )
    pre_grip_pose = StateController(
        stretch,
        {
            "wrist_roll_counterclockwise": 0.0,
            "gripper_open": 0.4,
        },
    )
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

    phase = "approach"
    in_zone = False
    previous_auto = set()
    print(f"Phase: {phase}")

    try:
        while True:
            command = {}

            if phase == "approach":
                success, frame, depth_frame = cameras.read_head()
                if success:
                    center = detect_object(frame)
                    if center:
                        x_pixel, y_pixel = center
                        width = frame.shape[1]
                        height = frame.shape[0]
                        horizontal_error = (x_pixel - width / 2) / (width / 2)
                        vertical_error = (y_pixel - height / 2) / (height / 2)
                        command["head_tilt_up"] = (
                            -KP_TILT * vertical_error
                        )
                        add_pose(command, head_forward)
                        command["base_counterclockwise"] = (
                            -KP_BASE_ALIGN * horizontal_error
                        )

                        camera_T = transforms.get_cam_T(cameras.head_info)
                        object_T = locate_object(
                            center, depth_frame, cameras.head_info, camera_T
                        )
                        if object_T is not None:
                            x, y, z = object_T[:3, 3]
                            angle = math.atan2(y, x)
                            distance = math.hypot(x, y)

                            if not in_zone and distance <= 0.55:
                                in_zone = True
                            elif in_zone and distance > 0.6:
                                in_zone = False

                            if in_zone:
                                command["head_pan_counterclockwise"] = (
                                    -KP_BASE_ALIGN * horizontal_error
                                )
                                angle_error = -math.pi / 2 - angle
                                command["base_counterclockwise"] = (
                                    -KP_ANGLE * angle_error
                                )
                                command["base_forward"] = 0.0
                                if abs(angle_error) < math.radians(5):
                                    phase = "align"
                                    cv2.destroyWindow("Head RGB")
                                    print(f"\nPhase: {phase}")
                            else:
                                alignment = 1.0 - abs(horizontal_error)
                                authority = np.clip(
                                    (alignment - 0.9) / 0.1, 0.0, 1.0
                                )
                                command["base_forward"] = (
                                    KP_FORWARD * distance * authority
                                )

                            wrist_T = transforms.get_cam_T(cameras.wrist_info)
                            command["lift_up"] = KP_LIFT * (
                                z - (wrist_T[2, 3] + 0.01) + 0.10
                            )

                        cv2.circle(frame, center, 5, (0, 255, 0), -1)

                    else:
                        add_pose(command, scanning_position)
                        command["base_counterclockwise"] = -0.3

                    if phase == "approach":
                        cv2.imshow("Head RGB", frame)

                add_pose(command, stow_pose)

            elif phase == "align":
                success, frame, depth_frame = cameras.read_wrist()
                if success:
                    center = detect_object(frame)
                    if center:
                        camera_T = transforms.get_cam_T(cameras.wrist_info)
                        object_T = locate_object(
                            center, depth_frame, cameras.wrist_info, camera_T
                        )
                        if object_T is not None:
                            x, y, z = object_T[:3, 3]
                            angle = math.atan2(y, x)
                            wrist_z = camera_T[2, 3]
                            angle_error = -math.pi / 2 - angle
                            lift_error = z - (wrist_z + 0.01)

                            command["base_counterclockwise"] = KP_ANGLE * (
                                -angle_error - math.radians(3)
                            )
                            command["lift_up"] = KP_LIFT * lift_error

                            if (
                                stow_pose.is_at_goal()
                                and abs(angle_error) < math.radians(5)
                                and abs(lift_error) < 0.03
                            ):
                                phase = "reach"
                                print(f"\nPhase: {phase}")

                        cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    cv2.imshow("Wrist RGB", frame)

                add_pose(command, stow_pose)

            elif phase == "reach":
                success, frame, depth_frame = cameras.read_wrist()
                if success:
                    center = detect_object(frame)
                    if center:
                        x_pixel, y_pixel = center
                        width = frame.shape[1]
                        height = frame.shape[0]
                        command["wrist_yaw_counterclockwise"] = (
                            KP_YAW * (x_pixel - width / 2) / width
                        )
                        command["wrist_pitch_up"] = (
                            -KP_PITCH * (y_pixel - height / 2) / height
                        )

                        distance = cameras.wrist_info.get_depth(
                            center, depth_frame, sample_radius=15
                        )
                        if distance is not None:
                            distance_error = distance - 0.12
                            command["arm_out"] = KP_ARM * distance_error
                            if (
                                abs(distance_error) < 0.02
                                and pre_grip_pose.is_at_goal()
                            ):
                                phase = "grab"
                                print(f"\nPhase: {phase}")

                        cv2.circle(frame, center, 5, (0, 255, 0), -1)
                    cv2.imshow("Wrist RGB", frame)

                add_pose(command, pre_grip_pose)

            elif phase == "grab":
                command["lift_up"] = 0.2
                command["gripper_open"] = -1.0

            command = {
                name: float(np.clip(velocity, -1.0, 1.0))
                for name, velocity in command.items()
                if velocity
            }
            for name in previous_auto - command.keys():
                command[name] = 0.0
            previous_auto = {
                name for name, velocity in command.items() if velocity != 0.0
            }

            command = teleop.get_manual_override(command)
            controller.set_command(command)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        controller.set_command(controller.zero_cmd)
        cameras.close()
        cv2.destroyAllWindows()
        stretch.stop()


if __name__ == "__main__":
    main()
