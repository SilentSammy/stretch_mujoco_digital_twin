from stretch_mujoco_api.sim_config import (
    ArucoCube,
    MeshObject,
    ObjectControlsConfig,
    RoboCasaConfig,
    RobotPose,
    SimConfig,
)

_DEFAULT = SimConfig(
    scene=None,
    robocasa=RoboCasaConfig(enabled=False),
    objects=[],
    robot_pose=RobotPose(
        position=(0.0, 0.0, 0.0),
        orientation=(1.0, 0.0, 0.0, 0.0),
    ),
    headless=False,
    show_viewer_ui=False,
    use_passive_viewer=True,
    camera_rate=30.0,
    timestep=None,
    object_controls=ObjectControlsConfig(enabled=True),
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

_ARUCO_GRAB_FROM_ABOVE = SimConfig(
    objects=[
        ArucoCube(
            name="aruco_0",
            marker_id=0,
            dictionary="DICT_5X5_50",
            position=(0.6, -0.025, 0.05),
            orientation=(0.5, 0.5, -0.5, -0.5),
            size=0.1,
            mass=0.1,
            gravity=False,
        ),
    ],

    robot_pose=RobotPose(
        position=(0.2, 0.0, 0.0),
        orientation=(0.7071, 0.0, 0.0, 0.7071),
    ),
)

_ANDROID_ARMY = SimConfig(
    objects=[
        MeshObject.from_folder(
            "stretch_mujoco/models/assets/custom_objects/"
            "mujoco_scanned_objects/models/Android_Lego",
            name=f"android_{row}_{column}",
            position=(0.6 + row * 0.2, -0.4 + column * 0.2, 0.02),
            scale=(0.5, 0.5, 0.5),
        )
        for row in range(5)
        for column in range(5)
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
        layout=1,
        style=1,
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
CONFIG = _ARUCO_GRAB_FROM_ABOVE
