import importlib.util
import platform

from .norm_vel_ctrl import NormVelController


IS_STRETCH_ENV = importlib.util.find_spec('stretch_body') is not None
IS_LINUX_ENV = platform.system() == 'Linux'


def __getattr__(name):
    if name == 'TeleopProvider':
        from .teleop_provider import TeleopProvider
        return TeleopProvider
    if name == 'Cameras':
        from .cameras import Cameras
        return Cameras
    raise AttributeError(name)

__all__ = [
    'NormVelController',
    'TeleopProvider',
    'Cameras',
    'IS_STRETCH_ENV',
    'IS_LINUX_ENV',
]
