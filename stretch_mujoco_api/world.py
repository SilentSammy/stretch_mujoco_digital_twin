"""Build a MuJoCo world from a scene-independent simulation configuration."""

from pathlib import Path

import mujoco
import numpy as np

from stretch_mujoco import utils

from .sim_config import MeshObject, SimConfig


class World:
    def __init__(self, config: SimConfig):
        self.config = config
        self._validate()

    def _validate(self):
        if self.config.robocasa.enabled and self.config.scene is not None:
            raise ValueError("Choose either a custom scene or RoboCasa, not both")
        if self.config.camera_rate <= 0:
            raise ValueError("camera_rate must be positive")
        if self.config.timestep is not None and self.config.timestep <= 0:
            raise ValueError("timestep must be positive")
        for obj in self.config.objects:
            if not obj.name:
                raise ValueError("Object names cannot be empty")
            if obj.mass is not None and obj.mass <= 0:
                raise ValueError(f"Object mass must be positive: {obj.name}")
            if obj.density <= 0:
                raise ValueError(f"Object density must be positive: {obj.name}")
            if any(scale <= 0 for scale in obj.scale):
                raise ValueError(f"Object scale must be positive: {obj.name}")
            if obj.collision_size and any(size <= 0 for size in obj.collision_size):
                raise ValueError(f"Collision size must be positive: {obj.name}")

    def simulator_kwargs(self):
        model, scene = self._build_model()
        pose = self.config.robot_pose
        return {
            "scene_xml_path": scene,
            "model": model,
            "camera_hz": self.config.camera_rate,
            "start_translation": list(pose.position) if pose else None,
            "start_rotation_quat": list(pose.orientation) if pose else None,
        }

    def _build_model(self):
        if self.config.robocasa.enabled:
            spec = self._load_robocasa()
            scene = None
        else:
            scene = self._resolve(self.config.scene or utils.default_scene_xml_path)
            if not self.config.objects and self.config.timestep is None:
                return None, str(scene)
            spec = mujoco.MjSpec.from_file(str(scene))
            scene = None

        names = set()
        for obj in self.config.objects:
            if obj.name in names:
                raise ValueError(f"Duplicate object name: {obj.name}")
            names.add(obj.name)
            self._add_object(spec, obj)

        model = spec.compile()
        if self.config.timestep is not None:
            model.opt.timestep = self.config.timestep
        return model, scene

    def _load_robocasa(self):
        from stretch_mujoco.robocasa_gen import model_generation_wizard

        robocasa = self.config.robocasa
        _, xml, _ = model_generation_wizard(
            task=robocasa.task,
            layout=robocasa.layout,
            style=robocasa.style,
        )
        return mujoco.MjSpec.from_string(xml)

    def _resolve(self, path: str | Path):
        path = Path(path)
        if not path.is_absolute():
            path = self.config._config_dir / path
        return path.resolve()

    def _add_object(self, spec: mujoco.MjSpec, obj: MeshObject):
        mesh_path = self._resolve(obj.mesh)
        if not mesh_path.is_file():
            raise FileNotFoundError(f"Object mesh not found: {mesh_path}")

        mesh_name = f"{obj.name}_mesh"
        generated_names = {
            obj.name,
            mesh_name,
            f"{obj.name}_texture",
            f"{obj.name}_material",
        }
        existing_names = {
            item.name
            for collection in (
                spec.bodies,
                spec.meshes,
                spec.textures,
                spec.materials,
            )
            for item in collection
        }
        duplicates = generated_names & existing_names
        if duplicates:
            raise ValueError(f"Object name conflicts with the scene: {sorted(duplicates)}")

        material_name = None
        spec.add_mesh(name=mesh_name, file=str(mesh_path), scale=obj.scale)
        collision_size, collision_position, mass = self._get_physics(obj, mesh_path)

        if obj.texture is not None:
            texture_path = self._resolve(obj.texture)
            if not texture_path.is_file():
                raise FileNotFoundError(f"Object texture not found: {texture_path}")
            texture_name = f"{obj.name}_texture"
            material_name = f"{obj.name}_material"
            spec.add_texture(
                name=texture_name,
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=str(texture_path),
            )
            spec.add_material(name=material_name, textures=["", texture_name])

        body = spec.worldbody.add_body(
            name=obj.name,
            pos=obj.position,
            quat=obj.orientation,
            gravcomp=0.0 if obj.gravity else 1.0,
        )
        body.add_freejoint()

        visual = {
            "name": f"{obj.name}_visual",
            "type": mujoco.mjtGeom.mjGEOM_MESH,
            "meshname": mesh_name,
        }
        if material_name:
            visual["material"] = material_name

        visual.update(contype=0, conaffinity=0, density=0)
        body.add_geom(
            name=f"{obj.name}_collision",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=collision_position,
            size=collision_size,
            mass=mass,
            rgba=(1.0, 0.0, 0.0, 0.0),
            friction=(1.0, 0.005, 0.0001),
        )

        body.add_geom(**visual)

    def _get_physics(self, obj: MeshObject, mesh_path: Path):
        if obj.collision_size is None:
            collision_size, inferred_position = self._get_mesh_bounds(
                mesh_path,
                obj.scale,
            )
        else:
            collision_size = obj.collision_size
            inferred_position = (0.0, 0.0, collision_size[2])

        collision_position = obj.collision_position or inferred_position
        volume = 8 * np.prod(collision_size)
        mass = obj.mass if obj.mass is not None else max(obj.density * volume, 0.01)
        return collision_size, collision_position, mass

    @staticmethod
    def _get_mesh_bounds(mesh_path: Path, scale):
        if mesh_path.suffix.lower() != ".obj":
            raise ValueError(
                f"Automatic collision geometry requires an OBJ mesh: {mesh_path}"
            )

        vertices = []
        with mesh_path.open(encoding="utf-8", errors="ignore") as mesh_file:
            for line in mesh_file:
                if line.startswith("v "):
                    vertices.append([float(value) for value in line.split()[1:4]])

        if not vertices:
            raise ValueError(f"Object mesh contains no vertices: {mesh_path}")

        vertices = np.asarray(vertices) * np.asarray(scale)
        minimum = vertices.min(axis=0)
        maximum = vertices.max(axis=0)
        size = (maximum - minimum) / 2
        if np.any(size <= 0):
            raise ValueError(f"Object mesh has zero-volume bounds: {mesh_path}")
        return tuple(size), tuple((minimum + maximum) / 2)
