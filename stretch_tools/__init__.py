import importlib.util
import platform

from .camera_info import (
    CamInfo,
    DepthCamInfo,
    HEAD_CAMERA,
    HEAD_DEPTH_CAMERA,
    HEAD_RGB_CAMERA,
    NAVIGATION_CAMERA,
    WRIST_CAMERA,
    WRIST_DEPTH_CAMERA,
    WRIST_RGB_CAMERA,
    close_cameras,
)
from .norm_vel_ctrl import NormVelController
from .lidar import LidarPlotter, filter_mast_points
from .state_control import StateController


IS_STRETCH_ENV = importlib.util.find_spec('stretch_body') is not None
IS_LINUX_ENV = platform.system() == 'Linux'


def __getattr__(name):
    if name == 'TeleopProvider':
        from .teleop_provider import TeleopProvider
        return TeleopProvider
    if name == 'Cameras':
        from .camera_info import Cameras
        return Cameras
    if name == 'RobotTransforms':
        from .robot_transforms import RobotTransforms
        return RobotTransforms
    if name == 'ObjectPlotter':
        from .object_plotter import ObjectPlotter
        return ObjectPlotter
    raise AttributeError(name)

__all__ = [
    'NormVelController',
    'filter_mast_points',
    'LidarPlotter',
    'TeleopProvider',
    'Cameras',
    'CamInfo',
    'DepthCamInfo',
    'HEAD_CAMERA',
    'HEAD_DEPTH_CAMERA',
    'HEAD_RGB_CAMERA',
    'WRIST_CAMERA',
    'WRIST_DEPTH_CAMERA',
    'WRIST_RGB_CAMERA',
    'NAVIGATION_CAMERA',
    'close_cameras',
    'StateController',
    'RobotTransforms',
    'ObjectPlotter',
    'IS_STRETCH_ENV',
    'IS_LINUX_ENV',
]
