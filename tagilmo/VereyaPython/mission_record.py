from typing import Optional

from .mission_record_spec import MissionRecordSpec, FrameRecordingSpec, FrameRecordingType
from .timestamped_video_frame import FrameType


class MissionRecord:
    def __init__(self, record: MissionRecordSpec):
        self.spec = record
        self.observations_path = ''
        self.rewards_path = ''
        self.commands_path = ''
        self.mission_ended_path = ''
        self.mp4_colourmap_path = ''
        if self.spec.isRecording():
            raise NotImplementedError('mission recording is not implemented')

    def isRecordingMP4(self, _type: FrameType) -> bool:
        it: Optional[FrameRecordingSpec] = self.spec.video_recordings.get(_type, None)
        return it is not None and it.fr_type == FrameRecordingType.VIDEO

    def isRecordingBmps(self, _type: FrameType) -> bool:
        it: Optional[FrameRecordingSpec] = self.spec.video_recordings.get(_type, None)
        return it is not None and it.fr_type == FrameRecordingType.BMP

    def isRecordingRewards(self) -> bool:
        return self.spec.is_recording_rewards

    def isRecordingObservations(self) -> bool:
        return self.spec.is_recording_observations

    def isRecordingCommands(self) -> bool:
        return self.spec.is_recording_commands

    def isRecording(self) -> bool:
        return self.spec.isRecording()

    def getObservationsPath(self) -> str:
        return self.observations_path

    def getRewardsPath(self) -> str:
        return self.rewards_path

    def getCommandsPath(self) -> str:
        return self.commands_path

    def getMissionEndedPath(self) -> str:
        return self.mission_ended_path

    def getMP4ColourMapPath(self) -> str:
        return self.mp4_colourmap_path
