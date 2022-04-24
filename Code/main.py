import logging
import project_constants
import time
from video_stabilization import stabilize_video


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


def main():
    logger = create_logger()
    logger.info("Welcome to our final project in video processing")

    "create time dictionary"
    logger.debug("Creating time dictionary")
    time_dictionary = create_time_dictionary()

    "video stabilization"
    logger.debug("Running video stabilization. input path : " + project_constants.INPUT_VIDEO_PATH)
    stabilization_start_time = time.time()
    stabilize_video(project_constants.INPUT_VIDEO_PATH, project_constants.OUTPUT_PATH)
    stabilization_end_time = time.time()
    stabilization_time = stabilization_end_time - stabilization_start_time
    logger.debug('video stabilization has been finished. Running time : ' +
                 str(stabilization_time) + ' seconds')
    time_dictionary['time_to_stabilize'] = stabilization_time


if __name__ == "__main__":
    main()
