import numpy.typing as npt
import struct
from dataclasses import dataclass
from enum import IntEnum
from .reward_xml import RewardXML


@dataclass(slots=True, frozen=True, init=False)
class TimestampedReward:
    reward: RewardXML
