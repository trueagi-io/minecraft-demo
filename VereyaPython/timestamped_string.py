from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TimestampedString:
    timestamp: int
    text: str
