import logging
import project_constants
import json

ID1 = 315488171
ID2 = 314756297


def create_time_dictionary():
    time_dictionary = {}.fromkeys(['time_to_stabilize', 'time_to_binary', 'time_to_alpha', 'time_to_matted',
                                   'time_to_output'], None)
    return time_dictionary


def create_general_logger():
    # Gets or creates a general logger
    log_format = '%(asctime)s : %(levelname)s : %(funcName)s : %(message)s'
    logging.basicConfig(filename=project_constants.LOGGER_NAME, level=logging.DEBUG, format=log_format)
    logger = logging.getLogger(project_constants.LOGGER_NAME)
    return logger


def write_to_json_file(file_path: str, dict_to_add: dict):
    with open(file_path, 'w') as f:
        json.dump(dict_to_add, f, indent=4)
