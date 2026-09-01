"""Small pyrealsense2-compatible adapter for simulated cameras."""

import threading
import time

import numpy as np

from stretch_mujoco.enums.stretch_cameras import StretchCameras


CAMERA_TIMEOUT = 1.0


class stream:
    color = "color"
    depth = "depth"


class format:
    bgr8 = "bgr8"
    z16 = "z16"


class camera_info:
    name = "name"
    serial_number = "serial_number"


class option:
    holes_fill = "holes_fill"


class _Filter:
    def set_option(self, _option, _value):
        pass

    def process(self, frame):
        return frame


class spatial_filter(_Filter):
    pass


class temporal_filter(_Filter):
    pass


class hole_filling_filter(_Filter):
    pass


class _Device:
    def __init__(self, name):
        self.name = name
        self.serial = f"SIM-{name}"

    def get_info(self, info):
        return self.name if info == camera_info.name else self.serial


_devices = [_Device("D435I"), _Device("D405")]
_sim = None
_last_access = {}
_registered = set()
_pending_removals = set()
_owners = {}
_claimed_devices = set()
_lock = threading.RLock()
_watchdog_stop = threading.Event()
_watchdog_thread = None


def _prepare_image(camera, image):
    if not camera.is_depth:
        return image
    return np.clip(image * 1000, 0, 65535).astype(np.uint16)


def _set_simulator(simulator):
    global _sim, _watchdog_thread
    _sim = simulator
    _watchdog_stop.clear()
    if _watchdog_thread is None or not _watchdog_thread.is_alive():
        _watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        _watchdog_thread.start()


def _clear_simulator(simulator):
    global _sim
    if _sim is not simulator:
        return
    _watchdog_stop.set()
    with _lock:
        for camera in list(_registered):
            _sim.remove_camera(camera)
        _registered.clear()
        _pending_removals.clear()
        _last_access.clear()
        _owners.clear()
        _claimed_devices.clear()
        _sim = None


def _watchdog():
    while not _watchdog_stop.wait(0.05):
        now = time.monotonic()
        with _lock:
            _pending_removals.update(
                camera
                for camera in _registered
                if now - _last_access.get(camera, 0.0) > CAMERA_TIMEOUT
            )
            if _pending_removals:
                camera = _pending_removals.pop()
                _registered.remove(camera)
                _last_access.pop(camera, None)
                _sim.remove_camera(camera)


def _add_owner(camera, owner):
    with _lock:
        _owners.setdefault(camera, set()).add(owner)


def _remove_owner(camera, owner):
    with _lock:
        owners = _owners.get(camera, set())
        owners.discard(owner)
        if owners:
            return
        _owners.pop(camera, None)
        if camera in _registered:
            _pending_removals.add(camera)


def _read(camera, return_data=False):
    retry_add = False
    while True:
        with _lock:
            _last_access[camera] = time.monotonic()
            _pending_removals.discard(camera)
            if camera not in _registered:
                _registered.add(camera)
                retry_add = True

        if retry_add:
            _sim.add_camera(camera)
            retry_add = False

        try:
            camera_data = _sim.pull_camera_data()
            image = camera_data.get_camera_data(
                camera,
                auto_rotate=False,
                auto_correct_rgb=True,
                use_depth_color_map=False,
            )
            image = _prepare_image(camera, image)
            return (image, camera_data) if return_data else image
        except ValueError:
            if not _sim.is_running():
                raise ConnectionError("The simulator stopped while waiting for a camera")
            retry_add = True
            time.sleep(0.1)


class context:
    def query_devices(self):
        return list(_devices)


class config:
    def __init__(self):
        self.serial = None
        self.streams = set()

    def enable_device(self, serial):
        self.serial = serial

    def enable_stream(self, stream_type, *args):
        self.streams.add(stream_type)


def _camera_for(device, stream_type):
    cameras = {
        ("D435I", stream.color): StretchCameras.cam_d435i_rgb,
        ("D435I", stream.depth): StretchCameras.cam_d435i_depth,
        ("D405", stream.color): StretchCameras.cam_d405_rgb,
        ("D405", stream.depth): StretchCameras.cam_d405_depth,
    }
    return cameras[(device.name, stream_type)]


class _Frame:
    def __init__(self, frames, camera):
        self.frames = frames
        self.camera = camera

    def get_data(self):
        camera_data = self.frames.data.get(self.camera)
        if camera_data is not None:
            try:
                image = camera_data.get_camera_data(
                    self.camera,
                    auto_rotate=False,
                    auto_correct_rgb=True,
                    use_depth_color_map=False,
                )
                with _lock:
                    _last_access[self.camera] = time.monotonic()
                    _pending_removals.discard(self.camera)
                image = _prepare_image(self.camera, image)
                return image
            except ValueError:
                pass

        image, camera_data = _read(self.camera, return_data=True)
        self.frames.data[self.camera] = camera_data
        return image


class _Frames:
    def __init__(self, cameras):
        self.cameras = cameras
        self.data = {}

    def get_color_frame(self):
        return _Frame(self, self.cameras[stream.color])

    def get_depth_frame(self):
        return _Frame(self, self.cameras[stream.depth])


class pipeline:
    def __init__(self):
        self.device = None
        self.cameras = {}

    def start(self, pipeline_config):
        if pipeline_config.serial is None:
            self.device = next(
                device for device in _devices if device.serial not in _claimed_devices
            )
        else:
            self.device = next(
                device for device in _devices if device.serial == pipeline_config.serial
            )

        _claimed_devices.add(self.device.serial)
        self.cameras = {
            stream_type: _camera_for(self.device, stream_type)
            for stream_type in pipeline_config.streams
        }
        for camera in self.cameras.values():
            _add_owner(camera, self)
        return self

    def wait_for_frames(self):
        return _Frames(self.cameras)

    def stop(self):
        for camera in self.cameras.values():
            _remove_owner(camera, self)
        if self.device is not None:
            _claimed_devices.discard(self.device.serial)


class VideoCapture:
    """Minimal cv2.VideoCapture equivalent for the navigation camera."""

    def __init__(self, _index):
        self.camera = StretchCameras.cam_nav_rgb
        _add_owner(self.camera, self)

    def read(self):
        return True, _read(self.camera)

    def release(self):
        _remove_owner(self.camera, self)
