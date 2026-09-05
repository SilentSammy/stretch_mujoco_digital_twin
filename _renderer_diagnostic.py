import mujoco
import numpy as np

model = mujoco.MjModel.from_xml_string('''<mujoco><worldbody>
<light pos="0 0 2"/>
<geom type="plane" size="2 2 .1" rgba=".7 .2 .1 1"/>
<camera name="test" pos="0 0 .5"/>
</worldbody></mujoco>''')
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

for safe in (False, True):
    failures = 0
    for cycle in range(10):
        old = mujoco.Renderer(model, height=32, width=32)
        live = mujoco.Renderer(model, height=32, width=32)
        live.enable_depth_rendering()
        live.update_scene(data, camera='test')
        before = live.render().copy()
        if safe:
            old._gl_context.make_current()
            old._mjr_context.free()
            old._mjr_context = None
        old.close()
        after = live.render()
        failures += not np.allclose(before, after)
        print('RENDER', safe, cycle, before.min(), after.min(), after.max(), flush=True)
        live.close()
    print('FAILURES', safe, failures, flush=True)
