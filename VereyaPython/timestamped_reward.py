import numpy.typing as npt
import struct
from dataclasses import dataclass
from enum import IntEnum
from .reward_xml import RewardXML


@dataclass(slots=True, init=True, frozen=True)
class TimestampedReward:
    reward: RewardXML
    timestamp: int


    def getAsSimpleString(self) -> str:
        res = ''
        for (k, v) in self.reward.reward_values.items():
            if res:
                res += ', '
            res += '{0}: {1}'.format(k, v)
        return res
