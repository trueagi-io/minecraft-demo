import asyncio



async def sendStringAndGetShortReply(ip_address: str, port: int, message: str):
    reader, writer = await asyncio.open_connection(ip_address, port)
    writer.write(message.encode())
    await writer.drain()

    data = await reader.read(1000)
    writer.close()
    await writer.wait_closed()
    return data.decode()
