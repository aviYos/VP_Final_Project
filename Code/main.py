import project_constants
import time
from video_stabilization import stabilize_video
import project_utils


def main():
    logger = project_utils.create_logger()
    logger.info("Welcome to our final project in video processing")

    "create time dictionary"
    logger.debug("Creating time dictionary")
    time_dictionary = project_utils.create_time_dictionary()

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
