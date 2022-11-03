"""
thin wrapper around tcp server
"""
import asyncio
import logging
from typing import Callable
from .tcp_server import TCPServer
from .timestamped_string import TimestampedString
from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector

logger = logging.getLogger()

class StringServer:
    def __init__(self,
                 io_service: object,
                 port: int,
                 handle_string: Callable[[TimestampedString], None],
                 log_name: str):
        self.io_service = io_service
        self.port = port
        self.handle_string = handle_string
        self.log_name = log_name
        self.server = None

    def start(self) -> None:
        self.server = TCPServer(self.io_service, self.port, self.__cb, self.log_name)
        fut = asyncio.run_coroutine_threadsafe(self.server.startAccept(), self.io_service)
        fut.add_done_callback(self.__log_server)
        fut.result()

    def __log_server(self, fut):
        if self.server and self.server.isRunning():
            logger.info('started string server %s on port %d', self.log_name, self.getPort())
        else:
            logger.warning('failed to start string server %s on port %d', self.log_name, self.getPort())

    def __cb(self, message: TimestampedUnsignedCharVector) -> None:
        string_message = TimestampedString.from_vector(message)
        self.handle_string(string_message)
        self.recordMessage(string_message)

    def recordMessage(self, message: TimestampedString) -> None:
        pass

    def getPort(self) -> int:
        return self.server.getPort()

    def close(self) -> None:
        self.server.close()

    def record(self, path: str) -> 'StringServer':
        raise NotImplementedError('recording in string server not implemented')
       # if self.writer.is_open():
       #     self.writer.close()

       # self.writer.open(path, 'aw')
       # return self
