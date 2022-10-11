from dataclasses import dataclass
from typing import TYPE_CHECKING, List


@dataclass(slots=True)
class WorldState:
    has_mission_begun: bool
    is_mission_running: bool
    number_of_video_frames_since_last_state: int
    number_of_rewards_since_last_state: int
    number_of_observations_since_last_state: int
    video_frames: List[TimestampedVideoFrame]
    video_frames_colourmap: List[TimestampedVideoFrame]
    rewards: List[TimestampedReward]
    observations: List[TimestampedString]
    mission_control_messages: List[TimestampedString]
    errors: List[TimestampedString]

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

