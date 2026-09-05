"""Automatic keyboard controls for movable simulator objects."""

import threading
import time

import stretch_tools.input as input


class ObjectManipulator:
    def __init__(self, world, config):
        self.world = world
        self.config = config
        self.active = False
        self._held = set()
        self._gamepad_buttons = set()
        self._gamepad_axes = {}
        self._toggle_down = False
        self._selected = 0
        self._moved_objects = []
        self._stop = threading.Event()
        self._thread = None

    @property
    def selected(self):
        objects = list(self.world.objects.values())
        return objects[self._selected % len(objects)] if objects else None

    def start(self):
        if not self.config.enabled or not self.world.objects:
            return
        input.set_input_interceptor(self._intercept)
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        input.clear_input_interceptor(self._intercept)
        if self._thread is not None:
            self._thread.join()

    def _intercept(self, source, key, value):
        key = key.lower()
        toggle = (
            self.config.toggle_key if source == "keyboard"
            else self.config.gamepad_toggle
        ).lower()
        if key == toggle:
            if value and not self._toggle_down:
                self.active = not self.active
                self._toggle_down = True
                self._held.clear()
                self._gamepad_buttons.clear()
                self._gamepad_axes.clear()
                if self.active:
                    self._moved_objects.clear()
                    input.release_inputs()
                else:
                    self._print_moved_poses()
                state = "ON" if self.active else "OFF"
                selected = f" ({self.selected.name})" if self.active else ""
                print(f"Object controls: {state}{selected}")
            elif not value:
                self._toggle_down = False
            return True

        if not self.active:
            return False

        if source == "gamepad":
            self._update_gamepad(key.upper(), value)
            return True

        if value and key not in self._held:
            if key == self.config.next_object_key.lower():
                self._select(1)
            elif key == self.config.previous_object_key.lower():
                self._select(-1)
            elif key == self.config.gravity_off_key.lower():
                self.selected.set_gravity(False)
                print(f"Gravity OFF: {self.selected.name}")
            elif key == self.config.gravity_on_key.lower():
                self.selected.set_gravity(True)
                print(f"Gravity ON: {self.selected.name}")

        if value:
            self._held.add(key)
        else:
            self._held.discard(key)
        return True

    def _update_gamepad(self, key, value):
        if key in ("LX", "LY", "RX", "RY", "LT", "RT"):
            self._gamepad_axes[key] = value
            return

        if value and key not in self._gamepad_buttons:
            if key == self.config.gamepad_next_object:
                self._select(1)
            elif key == self.config.gamepad_previous_object:
                self._select(-1)
            elif key == self.config.gamepad_gravity_off:
                self.selected.set_gravity(False)
                print(f"Gravity OFF: {self.selected.name}")
            elif key == self.config.gamepad_gravity_on:
                self.selected.set_gravity(True)
                print(f"Gravity ON: {self.selected.name}")

        if value:
            self._gamepad_buttons.add(key)
        else:
            self._gamepad_buttons.discard(key)

    def _select(self, direction):
        self._selected = (self._selected + direction) % len(self.world.objects)
        print(f"Selected object: {self.selected.name}")

    def _print_moved_poses(self):
        for name in self._moved_objects:
            obj = self.world.objects[name]
            position = ", ".join(f"{value:.6f}" for value in obj.position)
            orientation = ", ".join(f"{value:.6f}" for value in obj.orientation)
            print(f"{name}:")
            print(f"    position=({position})")
            print(f"    orientation=({orientation})")

    def _run(self):
        period = 1.0 / self.config.update_rate
        while not self._stop.wait(period):
            obj = self.selected
            if not self.active or obj is None:
                continue

            keys = self.config.keys
            value = lambda positive, negative: (
                int(keys[positive].lower() in self._held)
                - int(keys[negative].lower() in self._held)
            )
            distance = self.config.translation_speed * period
            angle = self.config.rotation_speed * period
            axis = lambda name: (
                self._gamepad_axes.get(name, 0.0)
                if abs(self._gamepad_axes.get(name, 0.0)) >= self.config.gamepad_deadzone
                else 0.0
            )
            rotating = axis(self.config.gamepad_rotation_modifier) > 0.5
            game_translation = (0.0, 0.0, 0.0) if rotating else (
                axis("LX"), -axis("LY"), -axis("RY")
            )
            game_rotation = (
                axis("RX"), -axis("RY"), axis("LX")
            ) if rotating else (0.0, 0.0, 0.0)
            combine = lambda keyboard, gamepad: max(-1.0, min(1.0, keyboard + gamepad))
            delta = (
                combine(value("x+", "x-"), game_translation[0]) * distance,
                combine(value("y+", "y-"), game_translation[1]) * distance,
                combine(value("z+", "z-"), game_translation[2]) * distance,
                combine(value("roll+", "roll-"), game_rotation[0]) * angle,
                combine(value("pitch+", "pitch-"), game_rotation[1]) * angle,
                combine(value("yaw+", "yaw-"), game_rotation[2]) * angle,
            )
            if any(delta):
                obj.move_by(*delta)
                if obj.name not in self._moved_objects:
                    self._moved_objects.append(obj.name)
