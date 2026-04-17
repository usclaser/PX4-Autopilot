#!/usr/bin/env python3
"""
PX4 keyboard teleop: switch to Manual mode and drive manual_control_setpoint (arrow keys).

Requires ROS 2 (rclpy), px4_msgs, and pynput:
  pip install pynput
  # px4_msgs from a colcon workspace that matches your PX4 version

Before running, source your ROS 2 overlay that provides px4_msgs, e.g.:
  source /opt/ros/$ROS_DISTRO/setup.bash
  source install/setup.bash   # if you built px4_msgs locally

Run:
  python3 px4_keyboard_manual_teleop.py

Controls (spacecraft Manual/direct mapping in SpacecraftRateControl):
  Arrow Up/Down     body X thrust (forward / back)
  Arrow Left/Right    body Y thrust (left / right)
  Z / X               yaw torque (left / right)
  M                   send Manual mode (VEHICLE_CMD_DO_SET_MODE)
  A / D               arm / disarm
  Space               zero all sticks
  Q or Ctrl+C         quit

Safety: only use with props unpowered or vehicle restrained. You are responsible for arming rules.
"""

from __future__ import annotations

import math
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy

try:
    from pynput import keyboard
except ImportError as e:
    raise SystemExit(
        "Missing dependency: pynput. Install with: pip install pynput\n" + str(e)
    ) from e

from px4_msgs.msg import VehicleCommand, ManualControlSetpoint


# PX4: px4_custom_mode.h — MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1, PX4_CUSTOM_MAIN_MODE_MANUAL = 1
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1.0
PX4_CUSTOM_MAIN_MODE_MANUAL = 1.0


class Px4KeyboardManualTeleop(Node):
    def __init__(self) -> None:
        super().__init__("px4_keyboard_manual_teleop")
        self._shutdown_requested = False

        self.declare_parameter("target_system", 1)
        self.declare_parameter("stick_gain", 0.85)  # max deflection [-1, 1]
        self.declare_parameter("cmd_topic_vehicle_command", "/fmu/in/vehicle_command")
        self.declare_parameter("cmd_topic_manual_control", "/fmu/in/manual_control_input")
        self.declare_parameter("publish_rate_hz", 50.0)

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self._pub_cmd = self.create_publisher(
            VehicleCommand,
            self.get_parameter("cmd_topic_vehicle_command").get_parameter_value().string_value,
            qos,
        )
        self._pub_manual = self.create_publisher(
            ManualControlSetpoint,
            self.get_parameter("cmd_topic_manual_control").get_parameter_value().string_value,
            qos,
        )

        self._target_system = (
            self.get_parameter("target_system").get_parameter_value().integer_value
        )
        self._stick_gain = self.get_parameter("stick_gain").get_parameter_value().double_value
        rate = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        period = 1.0 / max(rate, 1.0)

        self._lock = threading.Lock()
        self._pressed: set = set()

        self._timer = self.create_timer(period, self._on_timer)

        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

        self.get_logger().info(
            "Started. Press M for Manual mode, A to arm, D to disarm. Arrow keys + Z/X to drive. Q to quit."
        )

    def destroy_node(self) -> bool:
        try:
            self._listener.stop()
        except Exception:
            pass
        return super().destroy_node()

    def _on_press(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key is None:
            return
        try:
            if key == keyboard.Key.esc:
                self._shutdown_requested = True
                raise keyboard.Listener.StopException()
            if getattr(key, "char", None) in ("q", "Q"):
                self._shutdown_requested = True
                raise keyboard.Listener.StopException()
        except keyboard.Listener.StopException:
            raise
        except Exception:
            pass

        if key == keyboard.Key.space:
            with self._lock:
                self._pressed.clear()
            return

        if hasattr(key, "char") and key.char is not None:
            c = key.char.lower()
            if c == "m":
                self._send_manual_mode()
            elif c == "a":
                self._send_arm(True)
            elif c == "d":
                self._send_arm(False)

        with self._lock:
            self._pressed.add(key)

    def _on_release(self, key: keyboard.Key | keyboard.KeyCode | None) -> None:
        if key is None:
            return
        with self._lock:
            self._pressed.discard(key)

    def _sticks_from_keys(self) -> tuple[float, float, float]:
        """Returns (pitch, roll, yaw) in [-1, 1] for ManualControlSetpoint."""
        g = float(self._stick_gain)
        pitch = 0.0
        roll = 0.0
        yaw = 0.0

        with self._lock:
            keys = set(self._pressed)

        if keyboard.Key.up in keys:
            pitch += g
        if keyboard.Key.down in keys:
            pitch -= g
        if keyboard.Key.left in keys:
            roll -= g
        if keyboard.Key.right in keys:
            roll += g

        # yaw: Z / X (lowercase)
        for k in keys:
            if hasattr(k, "char") and k.char is not None:
                if k.char.lower() == "z":
                    yaw -= g
                elif k.char.lower() == "x":
                    yaw += g

        def clamp(x: float) -> float:
            return max(-1.0, min(1.0, x))

        return clamp(pitch), clamp(roll), clamp(yaw)

    def _stamp_us(self) -> int:
        return int(time.time() * 1e6) & 0xFFFFFFFFFFFFFFFF

    def _send_manual_mode(self) -> None:
        msg = VehicleCommand()
        msg.timestamp = self._stamp_us()
        msg.command = VehicleCommand.VEHICLE_CMD_DO_SET_MODE
        msg.param1 = MAV_MODE_FLAG_CUSTOM_MODE_ENABLED
        msg.param2 = PX4_CUSTOM_MAIN_MODE_MANUAL
        msg.param3 = 0.0
        msg.param4 = 0.0
        msg.param5 = 0.0
        msg.param6 = 0.0
        msg.param7 = 0.0
        msg.target_system = int(self._target_system)
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.confirmation = 0
        msg.from_external = True
        self._pub_cmd.publish(msg)
        self.get_logger().info("Sent VEHICLE_CMD_DO_SET_MODE -> Manual")

    def _send_arm(self, arm: bool) -> None:
        msg = VehicleCommand()
        msg.timestamp = self._stamp_us()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0 if arm else 0.0
        msg.param2 = 0.0
        msg.param3 = 0.0
        msg.param4 = 0.0
        msg.param5 = 0.0
        msg.param6 = 0.0
        msg.param7 = 0.0
        msg.target_system = int(self._target_system)
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.confirmation = 0
        msg.from_external = True
        self._pub_cmd.publish(msg)
        self.get_logger().info("Sent ARM" if arm else "Sent DISARM")

    def _on_timer(self) -> None:
        if getattr(self, "_shutdown_requested", False):
            rclpy.shutdown()
            return

        pitch, roll, yaw = self._sticks_from_keys()

        out = ManualControlSetpoint()
        out.timestamp = self._stamp_us()
        out.timestamp_sample = out.timestamp
        out.valid = True
        out.data_source = ManualControlSetpoint.SOURCE_UNKNOWN
        out.pitch = float(pitch)
        out.roll = float(roll)
        out.yaw = float(yaw)
        out.throttle = float("nan")
        out.flaps = float("nan")
        for name in ("aux1", "aux2", "aux3", "aux4", "aux5", "aux6"):
            setattr(out, name, float("nan"))
        out.sticks_moving = bool(
            math.hypot(pitch, roll) > 0.05 or abs(yaw) > 0.05
        )
        out.buttons = 0

        self._pub_manual.publish(out)


def main() -> None:
    rclpy.init()
    node = Px4KeyboardManualTeleop()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
