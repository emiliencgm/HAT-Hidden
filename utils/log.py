#!/usr/bin/python
import logging


def create_logger(name='output.log') -> logging.Logger:
    """
    Creates a logger with a stream handler and two file handlers.

    The stream handler prints to the screen depending on the value of `quiet`.
    One file handler (verbose.log) saves all logs, the other (quiet.log) only saves important info.

    :param name: Name the log.
    :return: The logger.
    """

    logger = logging.getLogger('final.log')
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # create file handler which logs even debug messages
    fh = logging.FileHandler(f'{name}')
    fh.setLevel(logging.DEBUG)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.ERROR)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def create_logger_emilien(name='output.log') -> logging.Logger:
    """
    Creates a logger with a stream handler and a file handler.

    - The stream handler prints errors to the screen.
    - The file handler saves all log levels.

    :param name: Log file name.
    :return: The logger instance.
    """

    logger = logging.getLogger(name)  # Ensure each logger is unique
    logger.setLevel(logging.DEBUG)  # Allow all log levels

    if not logger.handlers:  # Prevent duplicate handlers
        # Create file handler that logs everything
        fh = logging.FileHandler(name, mode='a')  # Append mode
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

        # Create console handler for errors
        ch = logging.StreamHandler()
        ch.setLevel(logging.ERROR)
        ch.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))

        # Add handlers
        logger.addHandler(fh)
        logger.addHandler(ch)

        # **Key fix: Prevent log propagation to root logger**
        logger.propagate = False

    return logger
