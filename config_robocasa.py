from stretch_mujoco_api.sim_config import RoboCasaConfig, SimConfig


CONFIG = SimConfig(
    robocasa=RoboCasaConfig(
        enabled=True,
        task="PnPCounterToCab",
        layout=0,
        style=0,
    ),
)
