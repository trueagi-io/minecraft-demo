"""
thin wrapper around tcp server
"""
import os
import os.path
import asyncio
from asyncio import AbstractEventLoop
import logging
from typing import Callable, Optional
from .tcp_server import TCPServer
from .timestamped_string import TimestampedString
from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector
from .timestamped_string_writer import TimestampedStringWriter
logger = logging.getLogger()

class StringServer:
    def __init__(self,
                 io_service: AbstractEventLoop,
                 port: int,
                 handle_string: Callable[[TimestampedString], None],
                 log_name: str):
        self.io_service = io_service
        self.port = port
        self.handle_string = handle_string
        self.log_name = log_name
        self.server = TCPServer(self.io_service, self.port, self.__cb, self.log_name)
        self.writer = None

    def start(self) -> None:
        fut = asyncio.run_coroutine_threadsafe(self.server.startAccept(), self.io_service)
        fut.add_done_callback(self.__log_server)
        fut.result()

    def __log_server(self, fut):
        if self.server and self.server.isRunning():
            logger.info('started string server %s on port %d', self.log_name, self.getPort())
        else:
            logger.warning('failed to start string server %s on port %d', self.log_name, self.getPort())

    def __cb(self, message: TimestampedUnsignedCharVector) -> Optional[str]:
        string_message = TimestampedString.from_vector(message)
        self.recordMessage(string_message)
        return self.handle_string(string_message)

    def recordMessage(self, message: TimestampedString) -> None:
        if self.writer is None:
            return
        if self.writer.is_open():
            self.writer.write(message)

    def getPort(self) -> int:
        return self.server.getPort()

    def close(self) -> None:
        self.server.close()

    def record(self, path: str) -> 'StringServer':
        if not os.path.exists(os.path.dirname(path)):
            os.makedirs(os.path.dirname(path))
        if self.writer is None:
            self.writer = TimestampedStringWriter()
        if self.writer.is_open():
            self.writer.close()
        self.writer.open(path, 'wt')
        return self

    def stopRecording(self) -> None:
        pass
