from enum import IntEnum
from typing import Dict
from dataclasses import dataclass
from .timestamped_video_frame import FrameType
from dataclasses import field


class FrameRecordingType(IntEnum):
    BMP = 0
    VIDEO = 1
 

@dataclass(frozen=False, slots=True, init=False)
class MissionRecordSpec:

    @dataclass(slots=True, frozen=True)
    class FrameRecordingSpec:
        fr_type: FrameRecordingType 
        mp4_bit_rate: int
        mp4_fps: int
        drop_input_frames: bool

    video_recordings: Dict[FrameType, FrameRecordingSpec] = field(default_factory=lambda: dict())
    is_recording_observations: bool = False
    is_recording_rewards: bool = False
    is_recording_commands: bool = False
    destination: str = ''

    def __init__(self, destination: str=None):
        self.setDestination(destination)

    def setDestination(self, destination: str) -> None:
        if destination:
            raise NotImplementedError("setting destination is not implemented")

    def recordMP4(self, frames_per_second: int, bit_rate: int) -> None:
        raise NotImplementedError("recording is not implemented")

