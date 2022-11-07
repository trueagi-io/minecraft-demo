import asyncio
from asyncio import exceptions
from asyncio import AbstractEventLoop, Server, Future
import logging
import random
import time
from typing import Callable, Optional
from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector


logger = logging.getLogger()


class TCPServer:
    def __init__(self,
                 io_service: AbstractEventLoop,
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
        self.server: Optional[Server] = None

        assert(not asyncio.iscoroutinefunction(callback))

    def start(self):
        asyncio.run_coroutine_threadsafe(self.startAccept(), self.io_service).result()

    async def startAccept(self) -> None:
        port = self.port
        if port == 0:
            # attempt to assign a port from a predefined range
            port_min = 10000
            port_max = 11000 # TODO: could be configurable
            while True:
                port = random.randint(port_min, port_max)
                try:
                    logger.info('starting sever with port %i', port)
                    self.server = await asyncio.start_server(self.__cb, None, port)
                    logger.info('ok')
                    self.port = port
                    return
                except OSError as e:
                    logger.exception(e)
                    continue
        else:
            self.server = await asyncio.start_server(self.__cb, None, self.port)

    async def __cb(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        while not self.closing:
            # read header
            try:
                data = await reader.readexactly(4)
                expected = int.from_bytes(data, byteorder='big', signed=False)
            except exceptions.IncompleteReadError as e:
                # there is nothing to read apparently
                await asyncio.sleep(2) 
                continue
            try:
                data = await reader.readexactly(expected)
                result = TimestampedUnsignedCharVector(data=data, timestamp=time.time())
            except exceptions.IncompleteReadError as e:
                logger.exception("expection reading from stream in " + self.log_name, e)
                await asyncio.sleep(1) 
                continue

            if self.confirm_with_fixed_reply:
                writer.write(self.fixed_reply)
                await writer.drain()

            # run in threadpool, who knows how fast is our callback
            fut = self.io_service.run_in_executor(None, lambda: self.onMessageReceived(result))
            fut.add_done_callback(self.__done)

    def __done(self, fut: Future) -> None:
        e = fut.exception()
        if e is not None:
            logger.exception(f"Error running callback in {self.log_name}", exc_info=e)

    def expectSizeHeader(self, expect_size_header: bool):
        pass

    def confirmWithFixedReply(self, reply: str) -> None:
        assert(isinstance(reply, str))
        self.confirm_with_fixed_reply = True
        self.fixed_reply = reply.encode()

    def close(self):
        self.closing = True
        assert self.server is not None
        self.server.close()

    def getPort(self) -> int:
        return self.port

    def isRunning(self) -> bool:
        return self.server is not None and self.server.is_serving()
