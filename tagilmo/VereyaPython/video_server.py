import asyncio
from asyncio import AbstractEventLoop
import logging
from typing import Callable
import numpy
from .tcp_server import TCPServer
from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector
from .timestamped_video_frame import Transform, FrameType, TimestampedVideoFrame

logger = logging.getLogger()


class VideoServer:
    def __init__(self, loop: AbstractEventLoop,
                 port: int,
                 channels: int, frametype: FrameType,
                 handle_frame: Callable[[TimestampedVideoFrame], None]):
        self.io_service = loop
        self.handle_frame = handle_frame
        self.channels = channels
        self.frametype = frametype
        self.received_frames = 0
        self.written_frames = 0
        self.queued_frames = 0
        self.transform = Transform.REVERSE_SCANLINE
        self.port = port
        self.writers = list()
        self.server = TCPServer(self.io_service, port=self.port, callback=self.__cb, log_name="video")

    def start(self) -> None:
        fut = asyncio.run_coroutine_threadsafe(self.server.startAccept(), self.io_service)
        fut.add_done_callback(self.__log_server)
        fut.result()

    def __log_server(self, fut):
        if self.server and self.server.isRunning():
            logger.info('started video server on port %d', self.getPort())
        else:
            logger.warn('failed to start video server on port %d', self.getPort())

    def __cb(self, message: TimestampedUnsignedCharVector) -> None:
        # if len(message.data) != (TimestampedVideoFrame.FRAME_HEADER_SIZE + self.width * self.height * self.channels):
        #     # comment from c++ code
        #     # Have seen this happen during stress testing when a reward packet from (I think) a previous mission arrives during the next
        #     # one when the same port has been reassigned. Could throw here but chose to silently ignore since very rare.
        #     raise RuntimeError("message size {0}, but expected {1}".format(len(message.data),
        #                 TimestampedVideoFrame.FRAME_HEADER_SIZE + self.width * self.height * self.channels))
        frame = TimestampedVideoFrame(message, self.frametype)
        self.received_frames += 1
        self.handle_frame(frame)

    def close(self):
        self.server.close()

    def startRecording(self) -> None:
        self.written_frames = 0
        self.queued_frames = 0
        self.received_frames = 0
        for writer in self.writers:
            writer.open()

    def getPort(self) -> int:
        return self.server.getPort()

    def getWidth(self) -> int:
        return self.width

    def getHeight(self) -> int:
        return self.height

    def getChannels(self) -> int:
        return self.channels

    def getFrameType(self) -> FrameType:
        return self.frametype

    def stopRecording(self) -> None:
        for writer in self.writers:
            if (writer.isOpen()):
                writer.close()
                self.written_frames += writer.getFrameWriteCount()
        self.writers.clear()

    def receivedFrames(self) -> int:
        return self.received_frames

    def writtenFrames(self) -> int:
        return self.written_frames

    def recordMP4(self, path: str, frames_per_second: int, bit_rate: int, drop_input_frames: bool) -> None:
        raise NotImplementedError("mp4 recording is not implemented")

    def recordBmps(self, path: str) -> None:
        raise NotImplementedError("Bmp recording is not implemented")
