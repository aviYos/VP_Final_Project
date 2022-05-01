import project_constants
import time
from video_stabilization import stabilize_video
import project_utils
from background_subtraction import background_subtractor

import os

ID1 = 315488171
ID2 = 314756297


def main():
    logger = project_utils.create_logger()
    logger.info("Welcome to our final project in video processing")

    "create time dictionary"
    logger.debug("Creating time dictionary")
    time_dictionary = project_utils.create_time_dictionary()

    "video stabilization"
    logger.debug("Running video stabilization. input path : " + project_constants.INPUT_VIDEO_PATH)
    stabilization_start_time = time.time()
    stabilize_video(project_constants.INPUT_VIDEO_PATH, project_constants.STABILIZE_PATH,
                    project_constants.WINDOW_SIZE_TAU,
                    project_constants.MAX_ITER_TAU, project_constants.NUM_LEVELS_TAU, start_rows=10, start_cols=2,
                    end_rows=30, end_cols=30)
    stabilization_end_time = time.time()
    stabilization_time = stabilization_end_time - stabilization_start_time
    logger.debug('video stabilization has been finished. Running time : ' +
                 str(stabilization_time) + ' seconds')
    time_dictionary['time_to_stabilize'] = stabilization_time


if __name__ == "__main__":
    Class = background_subtractor(project_constants.SECOND_INPUT_VIDEO_PATH, project_constants.Background_Subtraction_Alpha,
                                  project_constants.Background_Subtraction_T,
                                  project_constants.Background_Subtraction_Theta
                                  ).run_background_subtraction()
    main()
