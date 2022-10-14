import asyncio


class ClientConnection:
    def __init__(self, io_service: object, 
                 address: str, port: int):
        self.loop = io_service
        self.address = address
        self.port = port
        self.timeout = 60
        fut = asyncio.run_coroutine_threadsafe(asyncio.open_connection(self.address, self.port), self.loop)
        self.reader, self.writer = fut.result(self.timeout) 

    def getTimeout(self) -> int:
        """Get the request/reply timeout.
        returns The timeout delay in seconds.
        """
        return self.timeout

    def setTimeout(seconds: int) -> int:
        """Set the request/reply timeout.
        param seconds The timeout delay in seconds."""
        result = self.timeout
        self.timeout = seconds
        return result

    def send(self, message: str) -> None:
        """Sends a string over the open connection.
        param message The string to send. Will have newline appended if needed."""
        asyncio.run_coroutine_threadsafe(self.__send(message), self.loop)

    async def __send(self, message: str) -> None:
        writer.write(message.encode())
        await asyncio.wait_for(writer.drain(), self.timeout)

    def close(self) -> None:
        self.writer.close()
        asyncio.run_coroutine_threadsafe(self.writer.wait_closed(), self.loop)

