from stretch_mujoco_api.sim_config import (
    ArucoCube,
    Cube,
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

_CUBES = SimConfig(
    objects=[
        Cube(
            name="dynamic_gravity_collision",
            color=(1.0, 1.0, 1.0, 1.0),
            position=(0.50, -0.15, 0.05),
            size=0.05,
            gravity=True,
            physics="dynamic",
            collision=True,
        ),
        Cube(
            name="dynamic_no_gravity_collision",
            color=(1.0, 0.2, 0.2, 1.0),
            position=(0.50, -0.05, 0.05),
            size=0.05,
            gravity=False,
            physics="dynamic",
            collision=True,
        ),
        Cube(
            name="dynamic_no_gravity_no_collision",
            color=(1.0, 0.6, 0.2, 1.0),
            position=(0.50, 0.05, 0.05),
            size=0.05,
            gravity=False,
            physics="dynamic",
            collision=False,
        ),
        Cube(
            name="kinematic_collision",
            color=(1.0, 1.0, 0.2, 1.0),
            position=(0.60, -0.10, 0.05),
            size=0.05,
            gravity=False,
            physics="kinematic",
            collision=True,
        ),
        Cube(
            name="kinematic_no_collision",
            color=(0.2, 1.0, 0.2, 1.0),
            position=(0.60, 0.10, 0.05),
            size=0.05,
            gravity=False,
            physics="kinematic",
            collision=False,
        ),
        Cube(
            name="fixed_collision",
            color=(0.2, 0.5, 1.0, 1.0),
            position=(0.70, -0.10, 0.05),
            size=0.05,
            gravity=False,
            physics="fixed",
            collision=True,
        ),
        Cube(
            name="fixed_no_collision",
            color=(1.0, 0.2, 1.0, 1.0),
            position=(0.70, 0.10, 0.05),
            size=0.05,
            gravity=False,
            physics="fixed",
            collision=False,
        ),
    ],
)

_TEXTURED_CUBE = SimConfig(
    objects=[
        Cube(
            name="six_colors",
            position=(0.6, 0.0, 0.3),
            size=0.1,
            # Editable PNG: 256px square faces, arranged U D L / R F B.
            texture="stretch_mujoco/models/assets/custom_objects/cube_texture.png",
            physics="kinematic",
            collision=False,
        ),
    ],
)

_ARUCO_CUBES = SimConfig(
    objects=[
        ArucoCube(
            name="aruco_0",
            marker_id=0,
            dictionary="DICT_5X5_50",
            position=(0.65, -0.025, 0.05),
            orientation=(1.0, 0.0, 0.0, 0.0), # Straight
            size=0.1,
        ),
        ArucoCube(
            name="aruco_1_six_faces",
            marker_id=1,
            dictionary="DICT_5X5_50",
            position=(0.18, -0.55, 0.505),
            size=0.05,
            marker_faces=6,
        ),
    ],

    robot_pose=RobotPose(
        position=(0.05, 0.0, 0.0),
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
