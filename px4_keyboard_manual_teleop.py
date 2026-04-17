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

True open-loop (no commanded wrench unless you move sticks / keys):
  • FC must be in PX4 Manual (nav_state MANUAL), not Position/Hold/Mission/Offboard-position.
    If QGC still station-keeps, you are not in Manual — verify nav_state and control_position.
  • In Manual + spacecraft, centered inputs → zero thrust/torque from the manual/direct path.
  • Disarm (D) when not testing so outputs stay in the disarmed state; do not arm until you intend to fire.
  • Avoid a second manual source (QGC virtual stick, RC) fighting this node.

On Linux/Ubuntu terminals, arrow keys send escape sequences; the script disables TTY echo so you
do not see stray characters like ^[A. If anything still prints, run: stty sane

Controls (spacecraft Manual/direct mapping in SpacecraftRateControl):
  Arrow Up/Down     body X thrust (forward / back)
  Arrow Left/Right    body Y thrust (left / right)
  Z / X               yaw torque (left / right)
  M                   send Manual mode (VEHICLE_CMD_DO_SET_MODE)
  A / D               arm / disarm (default: same as `commander arm -f` / normal disarm — see parameters)
  Space               zero all sticks
  Q or Ctrl+C         quit

Arming uses VEHICLE_CMD_COMPONENT_ARM_DISARM with param2=21196 when force_arm is true (PX4
magic value used by `commander arm -f`). from_external is set false in that case so preflight
checks are skipped like the NSH shell command — you cannot run `commander` itself from the host.

Safety: only use with props unpowered or vehicle restrained. You are responsible for arming rules.
"""

from __future__ import annotations

import math
import os
import sys
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

# Commander.cpp: `commander arm -f` / `disarm -f` use param2 == 21196 (force, skip arming checks)
PX4_FORCE_ARM_DISARM_MAGIC = 21196.0


def _tty_echo_off() -> tuple[int, list] | None:
    """Stop the terminal from echoing arrow-key escape sequences (e.g. ^[A on Linux). pynput still receives keys."""
    if os.name != "posix" or not sys.stdin.isatty():
        return None
    import termios

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    new = termios.tcgetattr(fd)
    # lflags: disable echo (and control-char echo if available)
    new[3] &= ~termios.ECHO
    if hasattr(termios, "ECHOCTL"):
        new[3] &= ~termios.ECHOCTL
    termios.tcsetattr(fd, termios.TCSADRAIN, new)
    return (fd, old)


def _tty_restore(saved: tuple[int, list] | None) -> None:
    if saved is None:
        return
    import termios

    fd, old = saved
    try:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    except OSError:
        pass


class Px4KeyboardManualTeleop(Node):
    def __init__(self) -> None:
        super().__init__("px4_keyboard_manual_teleop")
        self._shutdown_requested = False

        self.declare_parameter("target_system", 1)
        self.declare_parameter("stick_gain", 0.85)  # max deflection [-1, 1]
        self.declare_parameter("cmd_topic_vehicle_command", "/fmu/in/vehicle_command")
        self.declare_parameter("cmd_topic_manual_control", "/fmu/in/manual_control_input")
        self.declare_parameter("publish_rate_hz", 50.0)
        # Match `commander arm -f` (VEHICLE_CMD_COMPONENT_ARM_DISARM + magic param2, from_external=false)
        self.declare_parameter("force_arm", True)
        self.declare_parameter("force_disarm", False)
        # Manual stick throttle when not using throttle for translation [-1, 1]; -1 = full down (no +Z thrust cmd).
        # Avoids NaN so arming checks / consumers see a defined idle input.
        self.declare_parameter("idle_throttle", -1.0)

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
        self._force_arm = self.get_parameter("force_arm").get_parameter_value().bool_value
        self._force_disarm = self.get_parameter("force_disarm").get_parameter_value().bool_value
        self._idle_throttle = self.get_parameter("idle_throttle").get_parameter_value().double_value
        rate = self.get_parameter("publish_rate_hz").get_parameter_value().double_value
        period = 1.0 / max(rate, 1.0)

        self._lock = threading.Lock()
        self._pressed: set = set()

        self._timer = self.create_timer(period, self._on_timer)

        self._listener = keyboard.Listener(on_press=self._on_press, on_release=self._on_release)
        self._listener.start()

        self.get_logger().info(
            "Started. M=Manual, A=arm (force=%s), D=disarm (force=%s). Arrows+Z/X. Q=quit."
            % (self._force_arm, self._force_disarm)
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
        """Same mechanism as `commander arm [-f]` / `disarm [-f]` (not subprocess — PX4 NSH only)."""
        force = self._force_arm if arm else self._force_disarm
        msg = VehicleCommand()
        msg.timestamp = self._stamp_us()
        msg.command = VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM
        msg.param1 = 1.0 if arm else 0.0
        msg.param2 = PX4_FORCE_ARM_DISARM_MAGIC if force else 0.0
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
        # Commander.cpp: arm(..., cmd.from_external || !forced) — for forced arm, from_external must
        # be false to skip preflight (matches internal send_vehicle_command used by NSH `commander`).
        msg.from_external = not force
        self._pub_cmd.publish(msg)
        tag = "ARM" if arm else "DISARM"
        if force:
            tag += " (force, param2=%d)" % int(PX4_FORCE_ARM_DISARM_MAGIC)
        self.get_logger().info("Sent %s" % tag)

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
        out.throttle = float(self._idle_throttle)
        out.flaps = float("nan")
        for name in ("aux1", "aux2", "aux3", "aux4", "aux5", "aux6"):
            setattr(out, name, float("nan"))
        out.sticks_moving = bool(
            math.hypot(pitch, roll) > 0.05 or abs(yaw) > 0.05
        )
        out.buttons = 0

        self._pub_manual.publish(out)


def main() -> None:
    tty_saved = _tty_echo_off()
    rclpy.init()
    node = None
    try:
        node = Px4KeyboardManualTeleop()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        _tty_restore(tty_saved)


if __name__ == "__main__":
    main()
