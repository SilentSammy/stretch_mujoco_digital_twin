import cv2

from stretch_tools import Cameras, NormVelController, TeleopProvider, input
from stretch_tools.cameras import (
    HEAD_COLOR,
    HEAD_DEPTH,
    WIDE_COLOR,
    WRIST_COLOR,
    WRIST_DEPTH,
)

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


def main():
    stretch = robot.Robot()
    if not stretch.startup():
        return

    cameras = Cameras()

    stretch.enable_collision_mgmt()
    controller = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)
    feeds = {
        "1": ("Head color", HEAD_COLOR),
        "2": ("Head depth", HEAD_DEPTH),
        "3": ("Wrist color", WRIST_COLOR),
        "4": ("Wrist depth", WRIST_DEPTH),
        "5": ("Wide color", WIDE_COLOR),
    }
    enabled = {key: False for key in feeds}
    open_windows = set()

    try:
        while True:
            command = teleop.get_manual_override({})
            controller.set_command(command)

            for key, (window, _) in feeds.items():
                if input.rising_edge(key):
                    enabled[key] = not enabled[key]
                    if not enabled[key] and window in open_windows:
                        cv2.destroyWindow(window)
                        open_windows.remove(window)

            for key, (window, feed) in feeds.items():
                if not enabled[key]:
                    continue
                success, frame = cameras.read(feed)
                if success:
                    if feed.endswith("depth"):
                        frame = cv2.applyColorMap(
                            cv2.convertScaleAbs(frame, alpha=0.03),
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
        cameras.close()
        cv2.destroyAllWindows()
        stretch.stop()


if __name__ == "__main__":
    main()
