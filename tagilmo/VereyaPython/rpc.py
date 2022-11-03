import sys
import logging
import asyncio


logger = logging.getLogger()


async def sendStringAndGetShortReply(ip_address: str, port: int, message: str, expect_size_header=True):
    logger.info('connecting to %s:%d', ip_address, port)
    reader, writer = await asyncio.open_connection(ip_address, port)
    logger.debug('sending to %s:%d \n %s', ip_address, port, message)
    writer.write(message.encode())
    await writer.drain()

    if expect_size_header:
        data = await reader.readexactly(4)
        expected = int.from_bytes(data, byteorder='big', signed=False)
        logger.debug('expected %d', expected)
        data = await reader.readexactly(expected)
        assert len(data) == expected
    else:
        data = await reader.readline()
    logger.debug('reply %s', data)
    writer.close()
    await writer.wait_closed()
    return data.decode(encoding='ascii')
