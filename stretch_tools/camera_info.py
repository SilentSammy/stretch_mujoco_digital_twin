import math
import importlib.util

import numpy as np


class CamInfo:
    def __init__(
        self,
        name,
        camera_matrix=None,
        distortion_coeffs=None,
        distortion_model=None,
        image_to_optical_T=None,
    ):
        self.name = name
        self.camera_matrix = camera_matrix
        self.distortion_coeffs = distortion_coeffs
        self.distortion_model = distortion_model
        self.image_to_optical_T = (
            np.eye(4) if image_to_optical_T is None else image_to_optical_T
        )

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
    ):
        super().__init__(
            name,
            camera_matrix,
            distortion_coeffs,
            distortion_model,
            image_to_optical_T,
        )
        self.depth_scale = depth_scale
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
        )

    def get_depth(self, pixel, depth_image, sample_radius=3):
        x_norm, y_norm = self.pixel_to_normalized(pixel)
        x = int(x_norm * self.depth_camera_matrix[0, 0] + self.depth_camera_matrix[0, 2])
        y = int(y_norm * self.depth_camera_matrix[1, 1] + self.depth_camera_matrix[1, 2])
        height, width = depth_image.shape[:2]
        region = depth_image[
            max(0, y - sample_radius) : min(height, y + sample_radius + 1),
            max(0, x - sample_radius) : min(width, x + sample_radius + 1),
        ]
        samples = region[region > 0]
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
)

WRIST_CAMERA = DepthCamInfo(
    name=f"D405 Wrist{_SIM_SUFFIX}",
    camera_matrix=WRIST_RGB_CAMERA.camera_matrix,
    depth_scale=1e-4 if _IS_STRETCH_ENV else 1e-3,
    distortion_coeffs=WRIST_RGB_CAMERA.distortion_coeffs,
    distortion_model=WRIST_RGB_CAMERA.distortion_model,
    depth_camera_matrix=WRIST_DEPTH_CAMERA.camera_matrix,
    depth_distortion_coeffs=WRIST_DEPTH_CAMERA.distortion_coeffs,
    depth_distortion_model=WRIST_DEPTH_CAMERA.distortion_model,
)

NAVIGATION_CAMERA = CamInfo(f"OV9782 Navigation{_SIM_SUFFIX}")
