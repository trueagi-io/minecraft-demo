from .mission_spec import MissionSpec
from .mission_record_spec import MissionRecordSpec
from .agent_host import AgentHost
from .client_info import ClientInfo
from .mission_exception import MissionException
from .mission_exception import MissionErrorCode
from .timestamped_string import TimestampedString
from .timestamped_video_frame import TimestampedVideoFrame, FrameType
import logging
import logging.handlers


ClientPool = set


def setupLogger():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)

    f = logging.handlers.RotatingFileHandler('app.log')
    f.setFormatter(formatter)
    f.setLevel(logging.DEBUG)
    logger.addHandler(f)
