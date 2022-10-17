import asyncio
import time
from typing import Callable
from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector


class TCPServer:
    def __init__(self,
                 io_service: object,
                 port: int,
                 callback: Callable[[TimestampedUnsignedCharVector], None],
                 log_name: str):
        self.io_service = io_service
        self.port = port
        self.onMessageReceived = callback
        self.confirm_with_fixed_reply = False
        self.fixed_reply = b''
        self.expect_size_header = True
        self.log_name = log_name
        self.closing = False
        self.server = None

        assert(not asyncio.iscoroutinefunction(callback))

        if port == 0:
            # attempt to assign a port from a predefined range
            port_min = 10000;
            port_max = 11000; # TODO: could be configurable
            self.bindToRandomPortInRange(io_service, port_min, port_max);

    def start(self):
        asyncio.run_coroutine_threadsafe(self.startAccept(), self.io_service).result()

    async def startAccept(self):
        self.server = await asyncio.start_server(self.__cb, None, self.port)

    async def __cb(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        while not self.closing:
            # read untile eof
            data = await reader.read()
            result = TimestampedUnsignedCharVector(data=data, timestamp=time.time())

            if self.confirm_with_fixed_reply:
                writer.write(self.fixed_reply)
                await writer.drain()

            # run in threadpool, who knows how fast is our callback
            await self.io_service.run_in_executor(None, lambda: self.onMessageReceived(result))

    def expectSizeHeader(self, expect_size_header: bool):
        pass

    def confirmWithFixedReply(self, reply: str) -> None:
        assert(isinstance(reply, str))
        self.confirm_with_fixed_reply = True

    def bindToPortInRange(self, port_min, port_max: int) -> None:
        import socket, errno

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

try:
    s.bind(("127.0.0.1", 5555))
except socket.error as e:
    if e.errno == errno.EADDRINUSE:
        print("Port is already in use")
    else:
        # something else raised the socket.error exception
        print(e)

s.close()
        self.fixed_reply = reply.encode()

    def close(self):
        self.closing = True
        self.server.close()

    def getPort(self) -> int:
        return self.port
