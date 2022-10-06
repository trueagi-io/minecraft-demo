

@dataclass(slots=True, frozen=True)
class TimestampedUnsignedCharVector:
    timestamp: int
    data: bytes
