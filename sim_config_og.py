from stretch_mujoco_api.sim_config import RoboCasaConfig, SimConfig


CONFIG = SimConfig(
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
