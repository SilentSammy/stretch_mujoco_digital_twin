import math
from pathlib import Path

import numpy as np
import stretch_urdf
from urchin import URDF

from .camera_info import HEAD_CAMERA, WRIST_CAMERA


class RobotTransforms:
    """URDF transforms calculated from the current Robot status."""

    def __init__(
        self,
        robot,
        model_name="SE3",
        tool_name="eoa_wrist_dw3_tool_sg3",
    ):
        self.robot = robot
        urdf_path = (
            Path(stretch_urdf.__path__[0])
            / model_name
            / f"stretch_description_{model_name}_{tool_name}.urdf"
        )
        self.urdf = URDF.load(str(urdf_path), lazy_load_meshes=True)

    def _get_joint_config(self):
        status = self.robot.get_status()
        arm_position = status["arm"]["pos"]
        head = status["head"]
        end_of_arm = status["end_of_arm"]

        return {
            "joint_head_pan": head["head_pan"]["pos"],
            "joint_head_tilt": head["head_tilt"]["pos"],
            "joint_lift": status["lift"]["pos"],
            "joint_arm_l0": arm_position / 4,
            "joint_arm_l1": arm_position / 4,
            "joint_arm_l2": arm_position / 4,
            "joint_arm_l3": arm_position / 4,
            "joint_wrist_yaw": end_of_arm["wrist_yaw"]["pos"],
            "joint_wrist_pitch": end_of_arm["wrist_pitch"]["pos"],
            "joint_wrist_roll": end_of_arm["wrist_roll"]["pos"],
        }

    def _get_link_transform(self, link):
        return self.urdf.link_fk(self._get_joint_config(), link=link)

    def get_head_cam_T(self, camera_info=HEAD_CAMERA):
        """Return the public head-image-frame-to-base transform."""
        urdf_T = self._get_link_transform("camera_color_optical_frame")
        return urdf_T @ camera_info.image_to_optical_T

    def get_wrist_cam_T(self, camera_info=WRIST_CAMERA):
        """Return the public wrist-image-frame-to-base transform."""
        urdf_T = self._get_link_transform("gripper_camera_color_optical_frame")
        return urdf_T @ camera_info.image_to_optical_T

    def get_base2world_T(self):
        """Return the base-to-world transform from planar odometry."""
        base = self.robot.get_status()["base"]
        x = base["x"]
        y = base["y"]
        theta = base["theta"]
        cos_theta = math.cos(theta)
        sin_theta = math.sin(theta)

        return np.array(
            [
                [cos_theta, -sin_theta, 0.0, x],
                [sin_theta, cos_theta, 0.0, y],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

    def get_cam_T(self, camera_info):
        """Return a camera-to-base transform for a camera-info object."""
        if "Head" in camera_info.name:
            return self.get_head_cam_T(camera_info)
        if "Wrist" in camera_info.name:
            return self.get_wrist_cam_T(camera_info)
        raise ValueError(f"Unknown camera: {camera_info.name}")

    def get_cam2world_T(self, camera_info):
        """Return a camera-to-world transform."""
        return self.get_base2world_T() @ self.get_cam_T(camera_info)
