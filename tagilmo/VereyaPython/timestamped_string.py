import time
from dataclasses import dataclass
from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector


@dataclass(slots=True, frozen=True)
class TimestampedString:
    timestamp: float
    text: str

    @staticmethod
    def from_vector(message: TimestampedUnsignedCharVector) -> 'TimestampedString':
        return TimestampedString(time.time(), message.data.decode())
