from enum import IntEnum, auto


class VideoPolicy(IntEnum):
    """Specifies what to do when there are more video frames being received than can be processed.
    """
    LATEST_FRAME_ONLY = auto() #  Discard all but the most recent frame. This is the default.
    KEEP_ALL_FRAMES = auto()           # Attempt to store all of the frames.


class RewardsPolicy(IntEnum):
    """ Specifies what to do when there are more rewards being received than can be processed.
    """
    LATEST_REWARD_ONLY = auto()      # Discard all but the most recent reward.
    SUM_REWARDS = auto()             # Add up all the rewards received. This is the default.
    KEEP_ALL_REWARDS = auto()        # Attempt to store all the rewards.

class ObservationsPolicy(IntEnum):
    """Specifies what to do when there are more observations being received than can be processed.
    """
    LATEST_OBSERVATION_ONLY = auto()   # Discard all but the most recent observation. This is the default.
    KEEP_ALL_OBSERVATIONS = auto()      # Attempt to store all the observations.


