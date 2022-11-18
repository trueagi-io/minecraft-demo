from dataclasses import dataclass
from .reward_xml import RewardXML


@dataclass(slots=True, init=True, frozen=True)
class TimestampedReward:
    reward: RewardXML
    timestamp: float


    def getAsSimpleString(self) -> str:
        res = ''
        for (k, v) in self.reward.reward_values.items():
            if res:
                res += ', '
            res += f'{k}: {v}'
        return res

    def add(self, other: 'TimestampedReward') -> None:
        for (k, v) in other.reward.reward_values.items():
            dimension: int = k
            value: float = v
            if dimension in self.reward.reward_values:
                self.reward.reward_values[dimension] += value
            else:
                self.reward.reward_values[dimension] = value

    @staticmethod
    def createFromSimpleString(timestamp: float, simple_string: str) -> 'TimestampedReward':
        reward = RewardXML()
        for item in simple_string.split(','):
            k, v = item.split(':')
            dimension = int(k)
            value = float(v)
            reward.reward_values[dimension] = value
        return TimestampedReward(timestamp=timestamp, reward=reward)
