"""Rendering regression test; requires an OpenGL-capable environment."""

import unittest

import mujoco
import numpy as np

from stretch_mujoco.mujoco_server_camera_manager import MujocoServerCameraManagerSync


class CameraRendererCleanupTest(unittest.TestCase):
    def test_closing_camera_preserves_other_cameras(self):
        model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
            <light pos="0 0 2"/>
            <geom type="plane" size="2 2 .1" rgba=".7 .2 .1 1"/>
            <camera name="test" pos="0 0 .5"/>
        </worldbody></mujoco>''')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        close = MujocoServerCameraManagerSync._close_camera_renderer

        for depth in (False, True):
            for cycle in range(10):
                with self.subTest(depth=depth, cycle=cycle):
                    old = mujoco.Renderer(model, height=32, width=32)
                    live = mujoco.Renderer(model, height=32, width=32)
                    try:
                        if depth:
                            live.enable_depth_rendering()
                        live.update_scene(data, camera="test")
                        before = live.render().copy()
                        self.assertTrue(np.any(before))
                        if depth:
                            np.testing.assert_allclose(before, 0.5, atol=1e-5)
                        # live's context is current when another camera expires.
                        close(old)
                        np.testing.assert_allclose(live.render(), before)
                    finally:
                        close(old)
                        close(live)


if __name__ == "__main__":
    unittest.main()
