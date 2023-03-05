from .timestamped_string import TimestampedString
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class TimestampedStringWriter:
    def __init__(self):
        self.file = None

    def open(self, path: str, mode: str) -> None:
        logger.info('opening %s for writing', path)
        self.file = open(path, mode)

    def write(self, message: TimestampedString) -> None:
        a = datetime.fromtimestamp(message.timestamp)
        self.file.write(f'{a.hour}:{a.minute}:{a.second}.{a.microsecond}')
        self.file.write(': ')
        self.file.write(message.text)
        self.file.write('\n')

    def close(self) -> None:
        self.file.close()
        self.file = None

    def is_open(self) -> bool:
        return self.file is not None