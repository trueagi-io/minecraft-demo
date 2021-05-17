import os
import logging
import logging.handlers


def setup_logger(log_file_path=None, level=logging.DEBUG, console_level=logging.INFO):
    log = logging.getLogger()
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] [%(threadName)s] %(message)s", "%H:%M:%S")
    console_handler.setFormatter(console_formatter)
    log.addHandler(console_handler)
    log.setLevel(logging.DEBUG)

    if log_file_path is not None:
        log_file_dir = os.path.dirname(log_file_path)
        if log_file_dir and not os.path.exists(log_file_dir):
            os.makedirs(log_file_dir)
        max_bytes = 1024 * 1024 * 20 # 20 mb
        handle = logging.handlers.RotatingFileHandler(log_file_path, mode='w', encoding='utf-8',
                maxBytes=max_bytes, backupCount=10)
        handle.setLevel(level)
        formatter = logging.Formatter("[%(asctime)s.%(msecs)03d] [%(levelname)-8s %(threadName)10s %(filename)24s:%(lineno)4d] %(message)s", "%d/%m/%Y %H:%M:%S")
        handle.setFormatter(formatter)
        log.addHandler(handle)





