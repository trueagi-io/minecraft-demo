import logging

def setup_logger(log_file='app.log'):
    # create logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s: %(levelname)-8s %(message)s')
    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    ch.setLevel(logging.INFO)
    # add ch to logger
    logger.addHandler(ch)


    f = logging.handlers.RotatingFileHandler(log_file)
    f.setFormatter(formatter)
    f.setLevel(logging.DEBUG)
    logger.addHandler(f)


