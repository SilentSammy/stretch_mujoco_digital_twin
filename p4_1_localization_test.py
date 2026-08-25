import time

import cv2
import numpy as np

from stretch_tools import (
    Cameras,
    HEAD_CAMERA,
    NormVelController,
    ObjectPlotter,
    RobotTransforms,
    TeleopProvider,
    WRIST_CAMERA,
)

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


CAMERA = WRIST_CAMERA  # Change to WRIST_CAMERA to use the wrist camera.
DEPTH_SAMPLE_RADIUS = 15


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
    depth = camera_info.get_depth(
        center, depth_frame, sample_radius=DEPTH_SAMPLE_RADIUS
    )
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

    controller = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)
    cameras = Cameras()
    if CAMERA is HEAD_CAMERA:
        read_camera = cameras.read_head
        camera_info = cameras.head_info
    else:
        read_camera = cameras.read_wrist
        camera_info = cameras.wrist_info
    transforms = RobotTransforms(stretch)
    plotter = ObjectPlotter()

    try:
        while plotter.is_open():
            controller.set_command(teleop.get_manual_override({}))

            success, frame, depth_frame = read_camera()
            camera_T = transforms.get_cam_T(camera_info)
            object_T = None
            if success:
                center = detect_object(frame)
                if center:
                    object_T = locate_object(
                        center, depth_frame, camera_info, camera_T
                    )
                    if object_T is not None:
                        cv2.circle(frame, center, 5, (0, 255, 0), -1)
                        position = object_T[:3, 3]
                        cv2.putText(
                            frame,
                            f"[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] m",
                            (center[0] + 10, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1,
                        )
                    else:
                        cv2.circle(frame, center, 5, (0, 255, 255), -1)
                        cv2.putText(
                            frame,
                            "No depth",
                            (center[0] + 10, center[1]),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )
                cv2.imshow(CAMERA.name, frame)

            plotter.update(camera_T, object_T)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        cameras.close()
        plotter.close()
        cv2.destroyAllWindows()
        stretch.stop()


if __name__ == "__main__":
    main()
