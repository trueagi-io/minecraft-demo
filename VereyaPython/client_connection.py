import asyncio
import logging


logger = logging.getLogger()


class ClientConnection:
    def __init__(self, io_service: object, 
                 address: str, port: int):
        self.loop = io_service
        self.address = address
        self.port = port
        self.timeout = 60
        fut = asyncio.run_coroutine_threadsafe(asyncio.open_connection(self.address, self.port), self.loop)
        logger.info(f'open command connection {self.address}:{self.port}')
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
        asyncio.run_coroutine_threadsafe(self.__send(message + '\n'), self.loop).add_done_callback(self.__on_done)

    def __on_done(self, fut) -> None:
        res = fut.result()
        e = fut.exception()
        if e is not None:
            logger.exception('error writing command', exec_info=e)

    async def __send(self, message: str) -> None:
        logger.info(f'writing command {message}')
        self.writer.write(message.encode())
        await asyncio.wait_for(self.writer.drain(), self.timeout)
        logger.info(f'done writing command {message}')

    def close(self) -> None:
        self.writer.close()
        asyncio.run_coroutine_threadsafe(self.writer.wait_closed(), self.loop)

