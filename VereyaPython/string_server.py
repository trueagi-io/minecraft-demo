"""
thin wrapper around tcp server
"""
from typing import Callable
from .tcp_server import TCPServer
from .timestamped_string import TimestampedString 
from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector


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
        self.server.start()

    def __cb(self, message: TimestampedUnsignedCharVector) -> None:
        string_message = TimestampedString.from_vector(message)
        self.handle_string(string_message)
        self.recordMessage(string_message)

    def recordMessage(self, message: TimestampedString) -> None:
        pass
 
    def getPort(self) -> int:
        return self.server.getPort()
