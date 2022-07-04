"""Collection of useful functions"""
from json.decoder import JSONDecodeError
from typing import Union
from pathlib import Path
import json
import logging
import time
from math import floor


def set_logging(log_file: Union[None, str], log_level: str, output_stdout: bool) -> None:
    """configures logging module.

    Args:
        log_file (str): path to log file. if omitted, logging will be forced to stdout.
        log_level (str): string name of log level (e.g. 'debug')
        output_stdout (bool): toggles stdout output. will be activated automatically if no log file was given.
            otherwise if activated, logging will be outputed both to stdout and log file.
    """
    logger = logging.getLogger()

    log_level_name = log_level.upper()
    log_level = getattr(logging, log_level_name)
    logger.setLevel(log_level)

    logging_format = logging.Formatter(
        '%(asctime)s - %(levelname)s [%(filename)s : %(funcName)s() : l. %(lineno)s]: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')

    if log_file:
        log_file_path = Path(log_file)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(logging_format)
        logger.addHandler(file_handler)
    else:
        output_stdout = True

    if output_stdout:
        stream = logging.StreamHandler()
        stream.setLevel(log_level)
        stream.setFormatter(logging_format)
        logger.addHandler(stream)


def check_paths(*args, names=None):
    """checks if the given paths exist.

    Args:
        args (list of paths): the paths to be checked

    Raises:
        ValueError: thrown if one of the given paths does not exist
    """
    for idx, path in enumerate(args):
        path = Path(path)
        if not path.exists():
            if names is not None and idx < len(names):
                raise ValueError(f"{str(names[idx])}: {str(path)} does not exist!")
            else:
                raise ValueError(f"file {str(path)} does not exist!")

def seconds_to_timestamp(seconds):
    """Converts the given seconds into a valid timestamp

    Args:
        seconds (float): the number of seconds

    Returns:
        str: the formated time string, format: hours:minutes:seconds
    """
    hours = floor(seconds / 3600)
    minutes = floor((seconds % 3600) / 60)
    seconds = round(seconds % 60, 3)
    return f"{str(hours)}h {str(minutes)}m {str(seconds)}s"

def read_json(path):
    """reads a json and converts it to dict

    Args:
        path (Path): path to json to be read

    Raises:
        JSONDecodeError: thrown if given json invalid

    Returns:
        dict: the parsed json fields
    """
    try:
        path = Path(path)
        with path.open() as jsonfile:
            content = json.load(jsonfile)
    except JSONDecodeError as e:
        logging.error(f"cannot parse json {path}: {e}")
        raise JSONDecodeError from e

    return content


class TimeMeasurement():
    """Wrapper for automatic time measurement. converted timestamp string can be outputed using custom functions."""

    def __init__(self, name, output_func=logging.debug):
        """initializes time measurement fields

        Args:
            name (str): name of function to measure time of. only used for output.
            output_func (callable, optional): function for time stamp output. Defaults to logging.debug.
        """
        self.name = name
        self.output_func = output_func
        self.start_time = 0

    def __enter__(self):
        """starts the timer when used in with-block"""
        self.start_time = time.time()

    def __exit__(self, *args):
        """outputs the time stamp"""
        diff = time.time() - self.start_time
        self.output_func(f"{self.name} took {seconds_to_timestamp(diff)} time!")

def fraction_to_percent(fraction, decimal_count=3):
    """converts a decimal number into percent representation. decimal places can be configured

    Args:
        fraction (float): number to convert into percentage
        decimal_count (int, optional): number of decimal places. Defaults to 3.

    Returns:
        float: percentage with given count of decimal places
    """
    return floor(fraction * 100 * (10 ** decimal_count)) / (10 ** decimal_count)

def file_size_megabytes(file_path):
    """computes the file size in megabytes

    Args:
        file_path (Path): path to file to compute the size of

    Returns:
        int: rounded file size in megabytes
    """
    file_path = Path(file_path)
    check_paths(file_path)

    return round(file_path.stat().st_size/1024**2)
