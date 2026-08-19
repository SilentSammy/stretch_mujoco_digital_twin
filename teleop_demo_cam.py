import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    WideCamera = cv2.VideoCapture
except ImportError:
    import stretch_mujoco_api.cameras as rs
    WideCamera = rs.VideoCapture

from stretch_tools import NormVelController, TeleopProvider, input

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


def main():
    stretch = robot.Robot()
    if not stretch.startup():
        return

    wrist = rs.pipeline()
    wrist_config = rs.config()
    wrist_device = next(
        device
        for device in rs.context().query_devices()
        if "D405" in device.get_info(rs.camera_info.name)
    )
    wrist_config.enable_device(wrist_device.get_info(rs.camera_info.serial_number))
    wrist_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 15)
    wrist_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    wrist.start(wrist_config)

    head = rs.pipeline()
    head_config = rs.config()
    head_config.enable_stream(rs.stream.color, 424, 240, rs.format.bgr8, 15)
    head_config.enable_stream(rs.stream.depth, 424, 240, rs.format.z16, 15)
    head.start(head_config)

    wide = WideCamera(6)

    stretch.enable_collision_mgmt()
    controller = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)
    feeds = {
        "1": "Head color",
        "2": "Head depth",
        "3": "Wrist color",
        "4": "Wrist depth",
        "5": "Wide color",
    }
    enabled = {key: False for key in feeds}

    try:
        while True:
            command = teleop.get_manual_override({})
            controller.set_command(command)

            for key, window in feeds.items():
                if input.rising_edge(key):
                    enabled[key] = not enabled[key]
                    if not enabled[key]:
                        cv2.destroyWindow(window)

            if enabled["1"] or enabled["2"]:
                head_frames = head.wait_for_frames()
                if enabled["1"]:
                    head_color = np.asanyarray(
                        head_frames.get_color_frame().get_data()
                    )
                    cv2.imshow(
                        "Head color",
                        cv2.rotate(head_color, cv2.ROTATE_90_CLOCKWISE),
                    )
                if enabled["2"]:
                    head_depth = np.asanyarray(
                        head_frames.get_depth_frame().get_data()
                    )
                    cv2.imshow(
                        "Head depth",
                        cv2.rotate(
                            cv2.applyColorMap(
                                cv2.convertScaleAbs(head_depth, alpha=0.03),
                                cv2.COLORMAP_JET,
                            ),
                            cv2.ROTATE_90_CLOCKWISE,
                        ),
                    )

            if enabled["3"] or enabled["4"]:
                wrist_frames = wrist.wait_for_frames()
                if enabled["3"]:
                    wrist_color = np.asanyarray(
                        wrist_frames.get_color_frame().get_data()
                    )
                    cv2.imshow("Wrist color", wrist_color)
                if enabled["4"]:
                    wrist_depth = np.asanyarray(
                        wrist_frames.get_depth_frame().get_data()
                    )
                    cv2.imshow(
                        "Wrist depth",
                        cv2.applyColorMap(
                            cv2.convertScaleAbs(wrist_depth, alpha=0.03),
                            cv2.COLORMAP_JET,
                        ),
                    )

            if enabled["5"]:
                _, wide_color = wide.read()
                cv2.imshow(
                    "Wide color",
                    cv2.rotate(wide_color, cv2.ROTATE_90_COUNTERCLOCKWISE),
                )
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

    except KeyboardInterrupt:
        pass
    finally:
        head.stop()
        wrist.stop()
        wide.release()
        cv2.destroyAllWindows()
        stretch.stop()


if __name__ == "__main__":
    main()
