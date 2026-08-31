"""Unitree wireless remote (gamepad) parser.

Byte layout copied verbatim from unitree_rl_gym's remote_controller.py so button
bits and stick floats decode identically on real hardware. Do NOT change offsets.
"""
import struct
import threading


class KeyMap:
    R1 = 0
    L1 = 1
    start = 2
    select = 3
    R2 = 4
    L2 = 5
    F1 = 6
    F2 = 7
    A = 8
    B = 9
    X = 10
    Y = 11
    up = 12
    right = 13
    down = 14
    left = 15


class RemoteController:
    def __init__(self):
        self.lx = 0.0
        self.ly = 0.0
        self.rx = 0.0
        self.ry = 0.0
        self.button = [0] * 16
        self._prev_button = [0] * 16
        self._pressed_latch = set()
        self._lock = threading.Lock()

    def set(self, data):
        # data == low_state.wireless_remote (40 bytes)
        keys = struct.unpack("H", data[2:4])[0]
        with self._lock:
            self._prev_button = self.button[:]
            for i in range(16):
                self.button[i] = (keys & (1 << i)) >> i
                if self.button[i] == 1 and self._prev_button[i] == 0:
                    self._pressed_latch.add(i)
            self.lx = struct.unpack("f", data[4:8])[0]
            self.rx = struct.unpack("f", data[8:12])[0]
            self.ry = struct.unpack("f", data[12:16])[0]
            self.ly = struct.unpack("f", data[20:24])[0]

    def is_pressed(self, key_idx):
        """Consume one latched rising-edge event for ``key_idx``."""
        with self._lock:
            if key_idx not in self._pressed_latch:
                return False
            self._pressed_latch.remove(key_idx)
            return True

    def is_down(self, key_idx):
        """Return the current level state without consuming an edge event."""
        with self._lock:
            return self.button[key_idx] == 1
