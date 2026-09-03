from stretch_mujoco_api.sim_config import (
    MeshObject,
    RoboCasaConfig,
    RobotPose,
    SimConfig,
)

_DEFAULT = SimConfig(
    scene=None,
    robocasa=RoboCasaConfig(enabled=False),
    objects=[],
    robot_pose=None,
    headless=False,
    show_viewer_ui=False,
    use_passive_viewer=True,
    camera_rate=30.0,
    timestep=None,
)

_OBJECT = SimConfig(
    objects=[
        MeshObject.from_folder(
            "stretch_mujoco/models/assets/custom_objects/"
            "mujoco_scanned_objects/models/Android_Lego",
            name="configured_lego",
            mesh_file="model.obj",
            texture_file="texture.png",
            position=(0.18, -0.55, 0.55),
            orientation=(1.0, 0.0, 0.0, 0.0),
            scale=(1.0, 1.0, 1.0),
            collision_size=(0.028, 0.020, 0.042),
            collision_position=(0.0, 0.0, 0.045),
            mass=0.1,
            density=1000.0,
            gravity=True,
        ),
    ],
)

_ROBOT_POSE = SimConfig(
    robot_pose=RobotPose(
        position=(0.5, 0.0, 0.0),
        orientation=(0.7071, 0.0, 0.0, 0.7071),
    ),
)

_ROBOCASA = SimConfig(
    robocasa=RoboCasaConfig(
        enabled=True,
        task="PnPCounterToCab",
        layout=0,
        style=0,
    ),
)

_ROBOCASA_OBJECT = SimConfig(
    robocasa=RoboCasaConfig(
        enabled=True,
        task="PnPCounterToCab",
        layout=0,
        style=0,
    ),
    objects=[
        MeshObject.from_folder(
            "stretch_mujoco/models/assets/custom_objects/"
            "mujoco_scanned_objects/models/Android_Lego",
            position=(0.0, 0.0, 1.0),
        ),
    ],
)

# Select the configuration to use.
CONFIG = _DEFAULT
