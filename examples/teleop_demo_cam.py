import cv2

from stretch_tools import (
    HEAD_CAMERA,
    NAVIGATION_CAMERA,
    WRIST_CAMERA,
    NormVelController,
    TeleopProvider,
    close_cameras,
    input,
)

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


def main():
    stretch = robot.Robot()
    if not stretch.startup():
        return

    stretch.enable_collision_mgmt()
    controller = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)
    feeds = {
        "1": ("Head color", HEAD_CAMERA.get_frame, False),
        "2": ("Head depth", HEAD_CAMERA.get_depth_frame, True),
        "3": ("Wrist color", WRIST_CAMERA.get_frame, False),
        "4": ("Wrist depth", WRIST_CAMERA.get_depth_frame, True),
        "5": ("Wide color", NAVIGATION_CAMERA.get_frame, False),
    }
    enabled = {key: False for key in feeds}
    open_windows = set()

    try:
        while True:
            command = teleop.get_manual_override({})
            controller.set_command(command)

            for key, (window, _, _) in feeds.items():
                if input.rising_edge(key):
                    enabled[key] = not enabled[key]
                    if not enabled[key] and window in open_windows:
                        cv2.destroyWindow(window)
                        open_windows.remove(window)

            for key, (window, get_frame, is_depth) in feeds.items():
                if not enabled[key]:
                    continue
                success, frame = get_frame()
                if success:
                    if is_depth:
                        frame = cv2.applyColorMap(
                            cv2.normalize(
                                frame, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U
                            ),
                            cv2.COLORMAP_JET,
                        )
                    scale = min(1, 480 / max(frame.shape[:2]))
                    frame = cv2.resize(frame, None, fx=scale, fy=scale)
                    cv2.imshow(window, frame)
                    open_windows.add(window)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    except KeyboardInterrupt:
        pass
    finally:
        close_cameras()
        cv2.destroyAllWindows()
        stretch.stop()


if __name__ == "__main__":
    main()
