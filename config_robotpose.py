from stretch_mujoco_api.sim_config import RobotPose, SimConfig


CONFIG = SimConfig(
    robot_pose=RobotPose(
        position=(0.5, 0.0, 0.0),
        orientation=(0.7071, 0.0, 0.0, 0.7071),
    ),
)
