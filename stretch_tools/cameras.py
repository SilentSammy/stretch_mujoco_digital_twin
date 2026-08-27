import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    WideCamera = cv2.VideoCapture
except ImportError:
    import stretch_mujoco_api.cameras as rs
    WideCamera = rs.VideoCapture

from .camera_info import HEAD_CAMERA, NAVIGATION_CAMERA, WRIST_CAMERA


HEAD_COLOR = "head_color"
HEAD_DEPTH = "head_depth"
WRIST_COLOR = "wrist_color"
WRIST_DEPTH = "wrist_depth"
WIDE_COLOR = "wide_color"


class Cameras:
    """Environment-independent camera access with uint16 millimetre depth."""

    def __init__(self, head_info=None, wrist_info=None, navigation_info=None):
        self._source_info = {
            "head": head_info or HEAD_CAMERA,
            "wrist": wrist_info or WRIST_CAMERA,
        }
        self.head_info = self._source_info["head"].with_depth_scale(1e-3)
        self.wrist_info = self._source_info["wrist"].with_depth_scale(1e-3)
        self.navigation_info = navigation_info or NAVIGATION_CAMERA
        self._pipelines = {}
        self._wide = None
        self._cache = {}

    def _start(self, name):
        if name in self._pipelines:
            return self._pipelines[name]

        model, width, height = (
            ("D435", 424, 240) if name == "head" else ("D405", 640, 480)
        )
        device = next(
            device
            for device in rs.context().query_devices()
            if model in device.get_info(rs.camera_info.name)
        )
        config = rs.config()
        config.enable_device(device.get_info(rs.camera_info.serial_number))
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 15)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 15)
        pipeline = rs.pipeline()
        pipeline.start(config)
        self._pipelines[name] = pipeline
        return pipeline

    def read(self, feed):
        if feed == WIDE_COLOR:
            try:
                if self._wide is None:
                    self._wide = WideCamera(6)
                success, frame = self._wide.read()
            except Exception:
                return False, None
            if success:
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            return success, frame

        if feed not in (HEAD_COLOR, HEAD_DEPTH, WRIST_COLOR, WRIST_DEPTH):
            raise ValueError(f"Unknown camera feed: {feed}")

        name = "head" if feed.startswith("head") else "wrist"
        stream = "color" if feed.endswith("color") else "depth"
        cache = self._cache.setdefault(name, {"frames": None, "used": set(), "time": 0})

        if (
            cache["frames"] is None
            or stream in cache["used"]
            or time.monotonic() - cache["time"] > 0.15
        ):
            try:
                cache["frames"] = self._start(name).wait_for_frames()
            except Exception:
                return False, None
            cache["used"].clear()
            cache["time"] = time.monotonic()

        try:
            frame = (
                cache["frames"].get_color_frame()
                if stream == "color"
                else cache["frames"].get_depth_frame()
            )
            image = np.asanyarray(frame.get_data())
        except Exception:
            cache["frames"] = None
            return False, None

        cache["used"].add(stream)
        if stream == "depth":
            scale = self._source_info[name].depth_scale * 1000
            image = np.clip(np.rint(image * scale), 0, 65535).astype(np.uint16)
        if name == "head":
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        return True, image

    def get_info(self, feed):
        if feed in (HEAD_COLOR, HEAD_DEPTH):
            return self.head_info
        if feed in (WRIST_COLOR, WRIST_DEPTH):
            return self.wrist_info
        if feed == WIDE_COLOR:
            return self.navigation_info
        raise ValueError(f"Unknown camera feed: {feed}")

    def read_head(self):
        color_ok, color = self.read(HEAD_COLOR)
        depth_ok, depth = self.read(HEAD_DEPTH)
        return color_ok and depth_ok, color, depth

    def read_wrist(self):
        color_ok, color = self.read(WRIST_COLOR)
        depth_ok, depth = self.read(WRIST_DEPTH)
        return color_ok and depth_ok, color, depth

    def close(self):
        for pipeline in self._pipelines.values():
            pipeline.stop()
        if self._wide is not None:
            self._wide.release()
        self._pipelines.clear()
        self._cache.clear()
        self._wide = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
