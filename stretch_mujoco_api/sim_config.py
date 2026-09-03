"""Configuration types for the simulated Stretch environment."""

from dataclasses import dataclass, field
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


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
class SimConfig:
    scene: str | Path | None = None
    robocasa: RoboCasaConfig = field(default_factory=RoboCasaConfig)
    objects: list[MeshObject] = field(default_factory=list)
    robot_pose: RobotPose | None = None
    headless: bool = False
    show_viewer_ui: bool = False
    use_passive_viewer: bool = True
    camera_rate: float = 30.0
    timestep: float | None = None
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
