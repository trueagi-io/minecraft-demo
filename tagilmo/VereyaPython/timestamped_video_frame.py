from typing import ClassVar
import struct
from enum import IntEnum
from dataclasses import dataclass
import logging

import numpy
import numpy as np
import numpy.typing as npt

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

    # check also Minecraft/src/main/java/com/microsoft/Malmo/Client/VideoHook.java
    FRAME_HEADER_SIZE: ClassVar[int] = 20 + (16 * 4 * 2)

    # The timestamp.
    timestamp: float

    # The width of the image in pixels.
    width: np.uint16

    #  The height of the image in pixels.
    height: np.uint16

    # The number of channels. e.g. 3 for RGB data, 4 for RGBD
    channels: np.uint8

    # The type of video data - eg 24bpp RGB, or 32bpp float depth
    frametype: FrameType

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

    # The pixels, stored as channels then columns then rows. Length should be width*height*channels.
    pixels: npt.NDArray[np.uint8] = numpy.array(0, dtype=numpy.uint8)

    def __init__(self, width: np.uint16, height: np.uint16,
                channels: np.uint8, message: TimestampedUnsignedCharVector,
                transform: Transform = Transform.IDENTITY,
                frametype: FrameType=FrameType.VIDEO):
        self.timestamp = message.timestamp
        self.width = width
        self.height = height
        self.channels = channels
        self.frametype = frametype

        # First extract the positional information from the header:
        self.xPos, self.yPos, self.zPos, self.yaw, self.pitch = struct.unpack('f' * 5, message.data[0: 5 * 4])
        pos = 5
        self.modelViewMatrix = numpy.frombuffer(message.data[pos * 4: (pos+16) * 4],
                                                dtype=np.dtype(numpy.float32), count=16)
        pos += 16
        self.calibrationMatrix = numpy.frombuffer(message.data[pos * 4: (pos+16) * 4],
                                                  dtype=np.dtype(numpy.float32), count=16)
        pos += 16
        assert (pos * 4) == self.FRAME_HEADER_SIZE
        stride = width * channels
        if transform == Transform.IDENTITY:
            self.pixels = numpy.frombuffer(message.data[self.FRAME_HEADER_SIZE:],
                                           dtype=np.dtype(numpy.uint8), count=stride * height)
        elif transform == Transform.RAW_BMP:
            self.pixels = numpy.frombuffer(message.data[self.FRAME_HEADER_SIZE:],
                                           dtype=np.dtype(numpy.uint8), count=stride * height)
            # Swap BGR -> RGB:
            for i in range(len(self.pixels) - 2):
                t = self.pixels[i]
                self.pixels[i] = self.pixels[i + 2]
                self.pixels[i + 2] = t
        elif transform == Transform.REVERSE_SCANLINE:
            self.pixels = numpy.zeros(int(stride) * int(height), dtype=numpy.uint8)
            offset = (height - 1) * stride
            start = 0
            for i in range(height):
                it = offset + self.FRAME_HEADER_SIZE
                self.pixels[start: start + stride] = numpy.frombuffer(message.data[it: it + stride],
                                                           dtype=np.dtype(numpy.uint8), count=stride)

                offset -= stride
                start += stride
        else:
            raise NotImplementedError(str(transform) + " is not implemented")
