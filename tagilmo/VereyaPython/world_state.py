from dataclasses import dataclass, field
from typing import List
from .video_server import TimestampedVideoFrame
from .timestamped_reward import TimestampedReward
from .timestamped_string import TimestampedString


@dataclass(slots=True, init=True)
class WorldState:
    has_mission_begun: bool = False
    is_mission_running: bool = False
    number_of_video_frames_since_last_state: int = 0
    number_of_rewards_since_last_state: int = 0
    number_of_observations_since_last_state: int = 0
    video_frames: List[TimestampedVideoFrame] = field(default_factory=list)
    video_frames_colourmap: List[TimestampedVideoFrame] = field(default_factory=list)
    video_frames_depth: List[TimestampedVideoFrame] = field(default_factory=list)
    rewards: List[TimestampedReward] = field(default_factory=list)
    observations: List[TimestampedString] = field(default_factory=list)
    mission_control_messages: List[TimestampedString] = field(default_factory=list)
    errors: List[TimestampedString] = field(default_factory=list)

    def clear(self):
        self.is_mission_running = False
        self.has_mission_begun = False
        self.number_of_observations_since_last_state = 0
        self.number_of_rewards_since_last_state = 0
        self.number_of_video_frames_since_last_state = 0
        self.observations.clear()
        self.rewards.clear()
        self.video_frames.clear()
        self.video_frames_colourmap.clear()
        self.mission_control_messages.clear()
        self.errors.clear()
