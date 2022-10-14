import asyncio
from typing import Callable
from .tcp_server import TCPServer
from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector
from .timestamped_video_frame import Transform, FrameType, TimestampedVideoFrame


class VideoServer:
    def __init__(self, loop: object, 
                 port: int,
                 width: int, height: int,
                 channels: int, frametype: FrameType,
                 handle_frame: Callable[[TimestampedVideoFrame], None]):
        self.io_service = loop
        self.handle_frame = handle_frame
        self.width = width
        self.height = height
        self.channels = channels
        self.frametype = frametype
        self.received_frames = 0
        self.written_frames = 0
        self.queued_frames = 0
        self.transform = Transform.REVERSE_SCANLINE
        self.port = port
        self.server = None

    def start(self) -> None:
        self.server = TCPServer(port=self.port, callback=self.__cb, log_name="video")
        asyncio.run_coroutine_threadsafe(self.server.startAccept(), self.io_service)

    def __cb(self, message: TimestampedUnsignedCharVector) -> None:
        if len(message.data) != (TimestampedVideoFrame.FRAME_HEADER_SIZE + self.width * self.height * self.channels):
            # comment from c++ code
            # Have seen this happen during stress testing when a reward packet from (I think) a previous mission arrives during the next
            # one when the same port has been reassigned. Could throw here but chose to silently ignore since very rare.
            raise RuntimeError("message size {0}, but expected {1}".format(len(message.data),
                        TimestampedVideoFrame.FRAME_HEADER_SIZE + self.width * self.height * self.channels))

        frame = TimestampedVideoFrame(self.width, self.height, self.channels, 
                                      message, self.transform, self.frametype)
        self.received_frames += 1
        self.handle_frame(frame)

    def close(self):
        self.server.close()
