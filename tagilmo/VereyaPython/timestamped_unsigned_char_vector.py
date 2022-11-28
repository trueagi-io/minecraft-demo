from dataclasses import dataclass

@dataclass(slots=True, frozen=True)
class TimestampedUnsignedCharVector:
    timestamp: float
    data: bytes
