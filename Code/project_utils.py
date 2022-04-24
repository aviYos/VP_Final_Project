import logging
import project_constants


def create_time_dictionary():
    time_dictionary = {}.fromkeys(['time_to_stabilize', 'time_to_binary', 'time_to_alpha', 'time_to_matted',
                                   'time_to_output'], None)
    return time_dictionary


def create_logger():
    # Gets or creates a logger
    log_format = '%(asctime)s : %(levelname)s : %(funcName)s : %(message)s'
    logging.basicConfig(filename=project_constants.LOGGER_NAME, level=logging.DEBUG, format=log_format)
    logger = logging.getLogger(project_constants.LOGGER_NAME)
    return logger
