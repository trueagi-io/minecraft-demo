from enum import IntEnum, auto


class MissionErrorCode(IntEnum):
    MISSION_BAD_ROLE_REQUEST = auto()
    MISSION_BAD_VIDEO_REQUEST = auto()
    MISSION_ALREADY_RUNNING = auto()
    MISSION_INSUFFICIENT_CLIENTS_AVAILABLE = auto()
    MISSION_TRANSMISSION_ERROR = auto()
    MISSION_SERVER_WARMING_UP = auto()
    MISSION_SERVER_NOT_FOUND = auto()
    MISSION_NO_COMMAND_PORT = auto()
    MISSION_BAD_INSTALLATION = auto()
    MISSION_CAN_NOT_KILL_BUSY_CLIENT = auto()
    MISSION_CAN_NOT_KILL_IRREPLACEABLE_CLIENT = auto()
    MISSION_VERSION_MISMATCH = auto()
        

class MissionException(RuntimeError):
    def __init__(self, message: str, code: MissionErrorCode):       
        super().__init__(message)
        self.code = code

