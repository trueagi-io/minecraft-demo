from enum import IntEnum
from dataclasses import dataclass
import logging
import json

import numpy as np
import numpy.typing as npt
import io
import os

from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector
logger = logging.getLogger()


class Transform(IntEnum):
    IDENTITY=0              # !< Don't alter the incoming bytes in any way
    RAW_BMP=1               # !< Layout bytes as raw BMP data (bottom-to-top RGB)
    REVERSE_SCANLINE=2      # !< Interpret input bytes as reverse scanline BGR


class FrameType(IntEnum):
    _MIN_FRAME_TYPE = 0
    VIDEO = _MIN_FRAME_TYPE     # !< Normal video, either 24bpp RGB or 32bpp RGBD
    DEPTH_MAP=1                 # !< 32bpp float depthmap
    LUMINANCE=2                 # !< 8bpp greyscale bitmap
    COLOUR_MAP=3                # !< 24bpp colour map
    _MAX_FRAME_TYPE=4

# should be frozen but init will be too ugly
@dataclass(slots=True, frozen=False, init=False)
class TimestampedVideoFrame:
    # camera to pixel opengl projection matrix
    calibrationMatrix: npt.NDArray[np.float32]
    modelViewMatrix: npt.NDArray[np.float32]

    # The timestamp.
    timestamp: float

    # The type of video data - eg 24bpp RGB, or 32bpp float depth
    frametype: FrameType

    # BMP image stored in bytes received by TCP from Vereya
    _pixels: bytes

    # The pitch of the player at render time
    pitch: float = 0

    # The yaw of the player at render time
    yaw: float = 0

    # The x pos of the player at render time
    xPos: float = 0

    # The y pos of the player at render time
    yPos: float = 0

    # The z pos of the player at render time
    zPos: float = 0

    iHeight: int = 0

    iWidth: int = 0

    iCh: int = 0

    def __init__(self, message: TimestampedUnsignedCharVector,
                 frametype: FrameType = FrameType.VIDEO):

        self.timestamp = message.timestamp
        self.frametype = frametype
        jo_len = int.from_bytes(message.data[0:4], byteorder='big', signed=False)
        json_string = message.data[4:jo_len+4].decode('utf-8')
        loadedjson = json.loads(json_string)
        self.xPos = loadedjson['x']
        self.yPos = loadedjson['y']
        self.zPos = loadedjson['z']
        self.yaw = loadedjson['yaw']
        self.pitch = loadedjson['pitch']
        self.iHeight = loadedjson['img_height'][0]
        self.iWidth = loadedjson['img_width'][0]
        self.iCh = loadedjson['img_ch'][0]
        self.modelViewMatrix = np.reshape(np.asarray(loadedjson['modelViewMatrix'], dtype=np.dtype(np.float32)), (4,4))

        self.calibrationMatrix = np.reshape(np.asarray(loadedjson['projectionMatrix'], dtype=np.dtype(np.float32)), (4,4))
        jo_len = jo_len + 4
        received_img_bytes = message.data[jo_len:]
        self._pixels = received_img_bytes

    @property
    def pixels(self):
        return np.flip(np.frombuffer(self._pixels, dtype="uint8").reshape((self.iHeight, self.iWidth, self.iCh))[:,:,:3],0)[..., ::-1].copy()
