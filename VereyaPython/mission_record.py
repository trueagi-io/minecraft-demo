from .mission_record_spec import MissionRecordSpec, FrameRecordingSpec, FrameRecordingType
from .timestamped_video_frame import FrameType


class MissionRecord:
    def __init__(self, record: MissionRecordSpec):
        self.spec = record

    def isRecordingMP4(self, _type: FrameType) -> bool:
        it: FrameRecordingSpec = self.spec.video_recordings.get(_type, None)
        return it is not None and it.fr_type == FrameRecordingType.VIDEO

    def isRecordingBmps(self, _type: FrameType) -> bool:
        it: FrameRecordingSpec = self.spec.video_recordings.get(_type, None)
        return it is not None and it.fr_type == FrameRecordingType.BMP

    def isRecordingRewards(self) -> bool: 
        return self.spec.is_recording_rewards

    def isRecordingObservations(self) -> bool: 
        return self.spec.is_recording_observations

    def isRecordingCommands(self) -> bool: 
        return self.spec.is_recording_commands

    def isRecording(self) -> bool:
        return self.spec.isRecording()

