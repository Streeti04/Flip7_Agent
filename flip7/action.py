from __future__ import annotations
from enum import IntEnum


class Action(IntEnum):
    STOP = 0
    DRAW = 1
    # Target selection for action cards — TARGET_i selects the i-th candidate
    # in the legal_actions list, NOT a player id. Up to 17 targets (18 players - self).
    TARGET_0 = 2
    TARGET_1 = 3
    TARGET_2 = 4
    TARGET_3 = 5
    TARGET_4 = 6
    TARGET_5 = 7
    TARGET_6 = 8
    TARGET_7 = 9
    TARGET_8 = 10
    TARGET_9 = 11
    TARGET_10 = 12
    TARGET_11 = 13
    TARGET_12 = 14
    TARGET_13 = 15
    TARGET_14 = 16
    TARGET_15 = 17
    TARGET_16 = 18
