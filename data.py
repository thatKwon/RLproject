from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np

class StatType(Enum):
    MAX_HP = auto()
    HP = auto()
    ATTACK = auto()
    ATTACK_RANGE = auto()
    ATTACK_SPEED = auto()
    DEFENSE = auto()
    MOVE_SPEED = auto()

@dataclass
class UnitStat:
    values: dict = field(default_factory=dict)

    def get(self, stat: StatType):
        return self.values.get(stat, 0.0)

    def set(self, stat: StatType, value: float):
        self.values[stat] = value

    def add(self, stat: StatType, value: float):
        self.values[stat] = self.values.get(stat, 0.0) + value

class UnitState(Enum):
    IDLE = auto()
    MOVE = auto()
    ATTACK = auto()
    DEAD = auto()

@dataclass
class Unit:
    team: int
    stats: UnitStat

    attack_target_idx: Optional[int] = None

    state: UnitState = UnitState.IDLE
    reaction_steps: int = 8

    pos: np.ndarray = field(default_factory=lambda: np.zeros(2))
    move_dir: np.ndarray = field(default_factory=lambda: np.zeros(2))