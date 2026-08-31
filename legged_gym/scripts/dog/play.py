#!/usr/bin/env python3
"""Run the interactive Isaac Gym demo for the custom dog backflip policy.

Example:
    python legged_gym/scripts/dog/play.py \
        --model=outputs/dog_backflip_run/stage1_nn/model_5000.pt

Controls:
    SPACE  trigger one backflip
    P      explicitly reset the robot and phase
    ESC    exit

The end of the phase does not reset physics. The policy remains in control of
landing, recovery and the following standing period.
"""

import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from legged_gym.envs.dog.play import main


if __name__ == "__main__":
    main()
