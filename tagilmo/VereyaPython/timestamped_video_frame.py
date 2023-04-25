from typing import ClassVar
import struct
from enum import IntEnum
from dataclasses import dataclass
import logging
import json

import numpy
import numpy as np
import numpy.typing as npt
import cv2
import io
import PIL.Image as Image

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
    # '12' for width, height and channels of the image
    # FRAME_HEADER_SIZE: ClassVar[int] = 20 + (16 * 4 * 2)
    # FRAME_HEADER_SIZE: ClassVar[int] = 84
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

    # def __init__(self, width: np.uint16, height: np.uint16,
    #             channels: np.uint8, message: TimestampedUnsignedCharVector,
    #             transform: Transform = Transform.IDENTITY,
    #             frametype: FrameType=FrameType.VIDEO):
    def __init__(self, message: TimestampedUnsignedCharVector,
                 transform: Transform = Transform.IDENTITY,
                 frametype: FrameType = FrameType.VIDEO):
        self.timestamp = message.timestamp
        # self.width = width
        # self.height = height
        # self.channels = channels
        self.frametype = frametype

        # header_length = len(message.data) - width*height*channels
        # header_data = message.data[:header_length]
        # image_data = message.data[header_length:]
        # decodedjson = header_data.decode('utf-8')
        # loadedjson = json.loads(decodedjson)
        # First extract the positional information from the header:
        jo_len = int.from_bytes(message.data[0:4], byteorder='big', signed=False)
        print("jo_len: " + str(jo_len) + "\n")
        json_string = message.data[4:jo_len+4].decode('utf-8')
        loadedjson = json.loads(json_string)
        self.xPos = loadedjson['x']
        self.yPos = loadedjson['y']
        self.zPos = loadedjson['z']
        self.yaw = loadedjson['yaw']
        self.pitch = loadedjson['pitch']
        self.width = loadedjson['width']
        self.height = loadedjson['height']
        # filepath = loadedjson['path_to_img']
        # self.pixels = cv2.imread(filepath[0])
        # self.pixels_rgb = cv2.cvtColor(self.pixels, cv2.COLOR_BGR2RGB)
        self.channels = loadedjson['channels']
        self.modelViewMatrix = np.reshape(np.asarray(loadedjson['modelViewMatrix'], dtype=np.dtype(numpy.float32)), (4,4))

        self.calibrationMatrix = np.reshape(np.asarray(loadedjson['projectionMatrix'], dtype=np.dtype(numpy.float32)), (4,4))
        jo_len = jo_len + 4
        received_png_bytes = message.data[jo_len:]
        f = io.BytesIO(received_png_bytes)
        png = Image.open(f)
        self.pixels = np.array(png)
        # jo_len = jo_len + 4

        # stride = self.width * self.channels
        # if transform == Transform.IDENTITY:
        #     self.pixels = numpy.frombuffer(message.data[jo_len:],
        #                                    dtype=np.dtype(numpy.uint8), count=stride * self.height)
        # elif transform == Transform.RAW_BMP:
        #     self.pixels = numpy.frombuffer(message.data[jo_len:],
        #                                    dtype=np.dtype(numpy.uint8), count=stride * self.height)
        #     # Swap BGR -> RGB:
        #     for i in range(len(self.pixels) - 2):
        #         t = self.pixels[i]
        #         self.pixels[i] = self.pixels[i + 2]
        #         self.pixels[i + 2] = t
        # elif transform == Transform.REVERSE_SCANLINE:
        #     self.pixels = numpy.zeros(int(stride) * int(self.height), dtype=numpy.uint8)
        #     offset = (self.height - 1) * stride
        #     start = 0
        #     for i in range(self.height):
        #         it = offset + jo_len
        #         self.pixels[start: start + stride] = numpy.frombuffer(message.data[it: it + stride],
        #                                                    dtype=np.dtype(numpy.uint8), count=stride)
        #
        #         offset -= stride
        #         start += stride
        # else:
        #     raise NotImplementedError(str(transform) + " is not implemented")
