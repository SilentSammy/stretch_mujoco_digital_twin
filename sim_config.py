from stretch_mujoco_api.sim_config import MeshObject, RoboCasaConfig, SimConfig

_OBJECT = SimConfig(
    objects=[
        MeshObject.from_folder(
            "stretch_mujoco/models/assets/custom_objects/"
            "mujoco_scanned_objects/models/Android_Lego",
            name="configured_lego",
            position=(0.18, -0.55, 0.55),
            collision_size=(0.028, 0.020, 0.042),
            collision_position=(0.0, 0.0, 0.045),
            mass=0.1,
        ),
    ],
)

CONFIG = SimConfig(
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
            name="configured_lego",
            position=(0.0, 0.0, 1.0),
            collision_size=(0.028, 0.020, 0.042),
            collision_position=(0.0, 0.0, 0.045),
            mass=0.1,
            gravity=False,
        ),
    ],
)
