import math
import importlib.util
import time

import cv2
import numpy as np

try:
    import pyrealsense2 as rs
    WideCamera = cv2.VideoCapture
except ImportError:
    import stretch_mujoco_api.cameras as rs
    WideCamera = rs.VideoCapture


HEAD_COLOR = "head_color"
HEAD_DEPTH = "head_depth"
WRIST_COLOR = "wrist_color"
WRIST_DEPTH = "wrist_depth"
WIDE_COLOR = "wide_color"


class CamInfo:
    def __init__(
        self,
        name,
        camera_matrix=None,
        distortion_coeffs=None,
        distortion_model=None,
        image_to_optical_T=None,
        feed=None,
    ):
        self.name = name
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.distortion_model = distortion_model
        self.feed = feed
        self.image_to_optical_T = (
            np.eye(4) if image_to_optical_T is None else image_to_optical_T
        )

    def get_frame(self):
        return _get_cameras().read(self.feed)

    @property
    def has_intrinsics(self):
        return self.camera_matrix is not None

    @property
    def fx(self):
        return self.camera_matrix[0, 0]

    @property
    def fy(self):
        return self.camera_matrix[1, 1]

    @property
    def cx(self):
        return self.camera_matrix[0, 2]

    @property
    def cy(self):
        return self.camera_matrix[1, 2]

    def pixel_to_normalized(self, pixel):
        return (pixel[0] - self.cx) / self.fx, (pixel[1] - self.cy) / self.fy

    def pixel_to_object_angles(self, pixel):
        x, y = self.pixel_to_normalized(pixel)
        return math.degrees(math.atan(y)), math.degrees(math.atan(x))

    def object_angles_to_pixel(self, yaw, pitch):
        x = math.tan(math.radians(yaw)) * self.fx + self.cx
        y = math.tan(math.radians(pitch)) * self.fy + self.cy
        return x, y


class DepthCamInfo(CamInfo):
    """Camera calibration where depth_scale is metres per depth-image unit."""

    def __init__(
        self,
        name,
        camera_matrix,
        depth_scale,
        distortion_coeffs=None,
        distortion_model=None,
        depth_camera_matrix=None,
        depth_distortion_coeffs=None,
        depth_distortion_model=None,
        image_to_optical_T=None,
        color_feed=None,
        depth_feed=None,
        native_depth_scale=None,
    ):
        super().__init__(
            name,
            camera_matrix,
            distortion_coeffs,
            distortion_model,
            image_to_optical_T,
            color_feed,
        )
        self.color_feed = color_feed
        self.depth_feed = depth_feed
        self.depth_scale = depth_scale
        self.native_depth_scale = (
            depth_scale if native_depth_scale is None else native_depth_scale
        )
        self.depth_camera_matrix = (
            camera_matrix if depth_camera_matrix is None else depth_camera_matrix
        )
        self.depth_distortion_coeffs = depth_distortion_coeffs
        self.depth_distortion_model = depth_distortion_model

    def with_depth_scale(self, depth_scale):
        return DepthCamInfo(
            self.name,
            self.camera_matrix,
            depth_scale,
            self.distortion_coeffs,
            self.distortion_model,
            self.depth_camera_matrix,
            self.depth_distortion_coeffs,
            self.depth_distortion_model,
            self.image_to_optical_T,
            self.color_feed,
            self.depth_feed,
            self.native_depth_scale,
        )

    def get_frames(self):
        cameras = _get_cameras()
        color_ok, color = cameras.read(self.color_feed)
        depth_ok, depth = cameras.read(self.depth_feed)
        return color_ok and depth_ok, color, depth

    def get_depth_frame(self):
        return _get_cameras().read(self.depth_feed)

    def get_depth(self, pixel, depth_image, sample_radius=3):
        x_norm, y_norm = self.pixel_to_normalized(pixel)
        x = int(x_norm * self.depth_camera_matrix[0, 0] + self.depth_camera_matrix[0, 2])
        y = int(y_norm * self.depth_camera_matrix[1, 1] + self.depth_camera_matrix[1, 2])
        height, width = depth_image.shape[:2]
        region = depth_image[
            max(0, y - sample_radius) : min(height, y + sample_radius + 1),
            max(0, x - sample_radius) : min(width, x + sample_radius + 1),
        ]
        samples = region[np.isfinite(region) & (region > 0)]
        if samples.size == 0:
            valid_y, valid_x = np.nonzero(
                np.isfinite(depth_image) & (depth_image > 0)
            )
            if valid_y.size == 0:
                return None
            distances = np.maximum(abs(valid_x - x), abs(valid_y - y))
            nearest = distances == distances.min()
            samples = depth_image[valid_y[nearest], valid_x[nearest]]
        return None if samples.size == 0 else float(np.median(samples) * self.depth_scale)


_IS_STRETCH_ENV = importlib.util.find_spec("stretch_body") is not None
_SIM_SUFFIX = "" if _IS_STRETCH_ENV else " (Sim)"
_HEAD_IMAGE_TO_OPTICAL_T = np.array(
    [
        [0.0, 1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

HEAD_RGB_CAMERA = CamInfo(
    name=f"D435i Head RGB{_SIM_SUFFIX}",
    feed=HEAD_COLOR,
    camera_matrix=np.array(
        [
            [303.07223511, 0.0, 122.78679657],
            [0.0, 303.06060791, 210.94392395],
            [0.0, 0.0, 1.0],
        ]
    ),
    distortion_coeffs=np.zeros(5),
    distortion_model="inverse_brown_conrady",
    image_to_optical_T=_HEAD_IMAGE_TO_OPTICAL_T,
)

HEAD_DEPTH_CAMERA = CamInfo(
    name=f"D435i Head Depth{_SIM_SUFFIX}",
    feed=HEAD_DEPTH,
    camera_matrix=np.array(
        [
            [214.76873779, 0.0, 120.41242218],
            [0.0, 214.76873779, 209.7878418],
            [0.0, 0.0, 1.0],
        ]
    ),
    distortion_coeffs=np.zeros(5),
    distortion_model="brown_conrady",
    image_to_optical_T=_HEAD_IMAGE_TO_OPTICAL_T,
)

WRIST_RGB_CAMERA = CamInfo(
    name=f"D405 Wrist RGB{_SIM_SUFFIX}",
    feed=WRIST_COLOR,
    camera_matrix=np.array(
        [
            [385.62329102, 0.0, 314.58789062],
            [0.0, 385.1807251, 243.30551147],
            [0.0, 0.0, 1.0],
        ]
    ),
    distortion_coeffs=np.array(
        [
            -0.0552569292,
            0.0598766357,
            -0.000858005136,
            -0.0000932277253,
            -0.0193387289,
        ]
    ),
    distortion_model="inverse_brown_conrady",
)

WRIST_DEPTH_CAMERA = CamInfo(
    name=f"D405 Wrist Depth{_SIM_SUFFIX}",
    feed=WRIST_DEPTH,
    camera_matrix=np.array(
        [
            [378.52832031, 0.0, 318.47045898],
            [0.0, 378.52832031, 241.03790283],
            [0.0, 0.0, 1.0],
        ]
    ),
    distortion_coeffs=np.zeros(5),
    distortion_model="brown_conrady",
)

HEAD_CAMERA = DepthCamInfo(
    name=f"D435i Head{_SIM_SUFFIX}",
    camera_matrix=HEAD_RGB_CAMERA.camera_matrix,
    depth_scale=1e-3,
    distortion_coeffs=HEAD_RGB_CAMERA.distortion_coeffs,
    distortion_model=HEAD_RGB_CAMERA.distortion_model,
    depth_camera_matrix=HEAD_DEPTH_CAMERA.camera_matrix,
    depth_distortion_coeffs=HEAD_DEPTH_CAMERA.distortion_coeffs,
    depth_distortion_model=HEAD_DEPTH_CAMERA.distortion_model,
    image_to_optical_T=HEAD_RGB_CAMERA.image_to_optical_T,
    color_feed=HEAD_COLOR,
    depth_feed=HEAD_DEPTH,
)

WRIST_CAMERA = DepthCamInfo(
    name=f"D405 Wrist{_SIM_SUFFIX}",
    camera_matrix=WRIST_RGB_CAMERA.camera_matrix,
    depth_scale=1e-3,
    distortion_coeffs=WRIST_RGB_CAMERA.distortion_coeffs,
    distortion_model=WRIST_RGB_CAMERA.distortion_model,
    depth_camera_matrix=WRIST_DEPTH_CAMERA.camera_matrix,
    depth_distortion_coeffs=WRIST_DEPTH_CAMERA.distortion_coeffs,
    depth_distortion_model=WRIST_DEPTH_CAMERA.distortion_model,
    color_feed=WRIST_COLOR,
    depth_feed=WRIST_DEPTH,
    native_depth_scale=1e-4 if _IS_STRETCH_ENV else 1e-3,
)

NAVIGATION_CAMERA = CamInfo(f"OV9782 Navigation{_SIM_SUFFIX}", feed=WIDE_COLOR)


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
        self._depth_filters = {}
        self._wide = None
        self._cache = {}

    def _get_depth_filters(self, name):
        if name not in self._depth_filters:
            spatial = rs.spatial_filter()
            temporal = rs.temporal_filter()
            hole_filling = rs.hole_filling_filter()
            hole_filling.set_option(rs.option.holes_fill, 1)
            self._depth_filters[name] = (spatial, temporal, hole_filling)
        return self._depth_filters[name]

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
            if stream == "depth":
                for depth_filter in self._get_depth_filters(name):
                    frame = depth_filter.process(frame)
            image = np.asanyarray(frame.get_data())
        except Exception:
            cache["frames"] = None
            return False, None

        cache["used"].add(stream)
        if stream == "depth":
            scale = self._source_info[name].native_depth_scale * 1000
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
        self._depth_filters.clear()
        self._cache.clear()
        self._wide = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


_CAMERAS = None


def _get_cameras():
    global _CAMERAS
    if _CAMERAS is None:
        _CAMERAS = Cameras()
    return _CAMERAS


def close_cameras():
    global _CAMERAS
    if _CAMERAS is not None:
        _CAMERAS.close()
        _CAMERAS = None
