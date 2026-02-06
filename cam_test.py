"""
Simple test: robot control with head camera display.
"""
from stretch_toolkit import controller, teleop, BACKEND_NAME, HEAD_CAMERA, WRIST_CAMERA, NAVIGATION_CAMERA
import time
import cv2

print(f"\n=== Running on {BACKEND_NAME} backend ===\n")

def teleop_demo():
    """Run teleoperation loop with head camera display."""
    print("Teleop with head camera view. Use gamepad/keyboard to control.")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            # Get normalized velocities from input devices
            velocities = teleop.get_normalized_velocities()
            
            # Send to robot (physical or simulated)
            controller.set_velocities(velocities)
            
            # Display head camera feed
            try:
                navigation_rgb = NAVIGATION_CAMERA.get_frame()
                cv2.imshow("Navigation Camera", navigation_rgb)
                cv2.waitKey(1)
            except:
                pass
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        # Stop all motion
        controller.set_velocities({})
        controller.stop()
        cv2.destroyAllWindows()
        print("Demo complete!")

if __name__ == "__main__":
    teleop_demo()
