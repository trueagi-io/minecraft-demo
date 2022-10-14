import asyncio
from typing import Callable
from .timestamped_unsigned_char_vector import TimestampedUnsignedCharVector


class TCPServer:
    def __init__(self,
                 port: int, 
                 callback: Callable[[TimestampedUnsignedCharVector], None],
                 log_name: str):
        self.onMessageReceived = callback
        self.confirm_with_fixed_reply = False
        self.fixed_reply = b''
        self.expect_size_header = True
        self.log_name = log_name
        self.closing = False

        assert(not asyncio.iscoroutinefunction(callback))

        if port == 0:
            raise NotImplementedError("port selection is not implemented")

    async def startAccept(self):
        self.server = await asyncio.start_server(self.__cb, None, self.port)

    async def __cb(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        while not self.closing:
            # read untile eof
            data = await reader.read()
            result = TimestampedUnsignedCharVector(data=data, timestampe=time.time())

            if self.confirm_with_fixed_repl:
                writer.write(self.fixed_reply)
                await writer.drain()

            loop = asyncio.get_running_loop()
            # run in threadpool, who knows how fast is our callback
            await loop.run_in_executor(None, lambda: self.callback(result))

    def expectSizeHeader(self, expect_size_header: bool):
        pass

    def confirmWithFixedReply(reply: str) -> None:
        assert(isinstance(reply, str))
        self.confirm_with_fixed_reply = True;
        self.fixed_reply = reply.encode()

    def close(self):
        self.closing = True
        self.server.close()
