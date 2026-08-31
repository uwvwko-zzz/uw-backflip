"""Initialize the Go2 low-level command used by backflip deployment."""
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo


def init_cmd_go(cmd: LowCmdGo, weak_motor=()):
    """Set the Go2 LowCmd header and per-motor mode flags. Must be called once before use."""
    cmd.head[0] = 0xFE
    cmd.head[1] = 0xEF
    cmd.level_flag = 0xFF
    cmd.gpio = 0
    PosStopF = 2.146e9
    VelStopF = 16000.0
    for i in range(len(cmd.motor_cmd)):
        # PMSM servo mode used by the official Go2 low-level examples.
        cmd.motor_cmd[i].mode = 0 if i in weak_motor else 0x01
        cmd.motor_cmd[i].q = PosStopF
        cmd.motor_cmd[i].dq = VelStopF
        cmd.motor_cmd[i].kp = 0
        cmd.motor_cmd[i].kd = 0
        cmd.motor_cmd[i].tau = 0
