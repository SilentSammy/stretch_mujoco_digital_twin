from stretch_tools import NormVelController, TeleopProvider

try:
    import stretch_body.robot as robot
except ImportError:
    import stretch_mujoco_api.robot as robot


def main():
    stretch = robot.Robot()
    stretch.startup()
    stretch.enable_collision_mgmt()
    
    controller = NormVelController(stretch)
    teleop = TeleopProvider(robot=stretch)

    try:
        while True:
            command = {}
            
            # Send velocity command with manual override
            command = teleop.get_manual_override(command)
            controller.set_command(command)

    except KeyboardInterrupt:
        pass
    finally:
        stretch.stop()

if __name__ == "__main__":
    main()
