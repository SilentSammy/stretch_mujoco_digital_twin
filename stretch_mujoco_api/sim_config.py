"""Configuration types for the simulated Stretch environment."""

from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


@dataclass
class RoboCasaConfig:
    enabled: bool = False
    task: str = "PnPCounterToCab"
    layout: int = 0
    style: int = 0


@dataclass
class RobotPose:
    position: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)


@dataclass
class MeshObject:
    name: str
    mesh: str | Path
    position: tuple[float, float, float]
    texture: str | Path | None = None
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    collision_size: tuple[float, float, float] | None = None
    collision_position: tuple[float, float, float] | None = None
    mass: float | None = None
    density: float = 1000.0
    gravity: bool = True
    physics: str = "dynamic"
    collision: bool = True

    @classmethod
    def from_folder(
        cls,
        folder: str | Path,
        *,
        name: str | None = None,
        mesh_file: str = "model.obj",
        texture_file: str | None = "texture.png",
        **kwargs,
    ):
        folder = Path(folder)
        return cls(
            name=name or folder.name.lower(),
            mesh=folder / mesh_file,
            texture=folder / texture_file if texture_file else None,
            **kwargs,
        )


@dataclass
class Cube:
    """Cube texture: a 2-row, 3-column atlas of square faces (U D L / R F B).

    Supply a filepath, a uint8 NumPy image (grayscale or OpenCV BGR/BGRA),
    or None for an untextured white cube.
    """

    name: str
    position: tuple[float, float, float]
    size: float = 0.1
    texture: str | Path | np.ndarray | None = None
    orientation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    mass: float | None = None
    density: float = 1000.0
    gravity: bool = True
    physics: str = "dynamic"
    collision: bool = True
    color: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)

    def get_texture(self):
        return self.texture


@dataclass(kw_only=True)
class ArucoCube(Cube):
    marker_id: int
    dictionary: str = "DICT_4X4_50"
    texture_size: int = 512
    marker_faces: int = 1
    texture: np.ndarray | None = field(default=None, init=False, repr=False)

    def get_texture(self):
        import cv2

        if self.marker_faces not in (1, 6):
            raise ValueError("marker_faces must be 1 or 6; use Cube for a plain cube")
        if self.texture_size <= 0:
            raise ValueError("texture_size must be positive")
        dictionary_id = getattr(cv2.aruco, self.dictionary, None)
        if dictionary_id is None:
            raise ValueError(f"Unknown ArUco dictionary: {self.dictionary}")
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        marker_size = round(self.texture_size * 0.8)
        marker = cv2.aruco.generateImageMarker(dictionary, self.marker_id, marker_size)
        face = np.full((self.texture_size, self.texture_size), 255, dtype=np.uint8)
        margin = (self.texture_size - marker_size) // 2
        face[margin:margin + marker_size, margin:margin + marker_size] = marker
        if self.marker_faces == 6:
            return np.tile(face, (2, 3))
        atlas = np.full((self.texture_size * 2, self.texture_size * 3), 255, dtype=np.uint8)
        # The F atlas face maps to the box's local +Z (top).
        atlas[self.texture_size:, self.texture_size:2 * self.texture_size] = face
        return atlas


@dataclass
class ObjectControlsConfig:
    enabled: bool = True
    toggle_key: str = "t"
    next_object_key: str = "m"
    previous_object_key: str = "n"
    gravity_off_key: str = "g"
    gravity_on_key: str = "f"
    gamepad_toggle: str = "START"
    gamepad_next_object: str = "RB"
    gamepad_previous_object: str = "LB"
    gamepad_gravity_off: str = "Y"
    gamepad_gravity_on: str = "B"
    gamepad_rotation_modifier: str = "LT"
    gamepad_deadzone: float = 0.15
    update_rate: float = 20.0
    translation_speed: float = 0.25
    rotation_speed: float = 1.0
    keys: dict[str, str] = field(default_factory=lambda: {
        "x+": "d",
        "x-": "a",
        "y+": "w",
        "y-": "s",
        "z+": "z",
        "z-": "x",
        "roll+": "u",
        "roll-": "o",
        "pitch+": "i",
        "pitch-": "k",
        "yaw+": "j",
        "yaw-": "l",
    })


@dataclass
class SimConfig:
    scene: str | Path | None = None
    robocasa: RoboCasaConfig = field(default_factory=RoboCasaConfig)
    objects: list[MeshObject | Cube] = field(default_factory=list)
    robot_pose: RobotPose | None = None
    headless: bool = False
    show_viewer_ui: bool = False
    use_passive_viewer: bool = True
    camera_rate: float = 30.0
    timestep: float | None = None
    object_controls: ObjectControlsConfig = field(default_factory=ObjectControlsConfig)
    _config_dir: Path = field(default_factory=Path.cwd, repr=False)


def load_sim_config(path: str | Path | None = None) -> SimConfig:
    if path is None:
        candidates = [
            Path.cwd() / "sim_config.py",
            Path(__file__).resolve().parents[1] / "sim_config.py",
        ]
        path = next((candidate for candidate in candidates if candidate.exists()), None)
        if path is None:
            return SimConfig()

    path = Path(path).resolve()
    spec = spec_from_file_location("_stretch_user_sim_config", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load simulator configuration from {path}")

    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    config = getattr(module, "CONFIG", None)
    if not isinstance(config, SimConfig):
        raise TypeError(f"{path} must define CONFIG as a SimConfig instance")

    config._config_dir = path.parent
    return config
