import project_constants
import time
from gaussian_stabilization import gaussian_stabilization
from background_subtraction import background_subtractor
from tracking import tracking
import project_utils

ID1 = 315488171
ID2 = 314756297


def main():
    logger = project_utils.create_general_logger()
    logger.info("Welcome to our final project in video processing")

    # Creating a time dictionary
    logger.debug("Creating time dictionary")
    time_dictionary = project_utils.create_time_dictionary()

    # Video Stabilization
    logger.debug("Running video stabilization. Input path: " + project_constants.INPUT_VIDEO_PATH)
    print('Stabilizing video:')
    start_time = time.time()
    gaussian_stabilization(project_constants.INPUT_VIDEO_PATH, project_constants.STABILIZE_PATH,
                           project_constants.motion)
    stabilization_end_time = time.time()
    stabilization_time = stabilization_end_time - start_time
    logger.debug('video stabilization has been finished. Running time : ' +
                 str(stabilization_time) + ' seconds')
    time_dictionary['time_to_stabilize'] = stabilization_time

    # Background Stabilization and Matting
    # Class = background_subtractor(project_constants.STABILIZED_VIDEO_PATH,
    #                              project_constants.Background_Subtraction_Alpha,
    #                              project_constants.Background_Subtraction_T,
    #                              project_constants.Background_Subtraction_Theta
    #                              ).run_background_subtraction()
    # Tracking
    logger.debug("Running tracking program. Matted path: " + project_constants.MATTED_PATH)
    tracking_start_time = time.time()
    # tracking(project_constants.MATTED_PATH, project_constants.BINARY_PATH, project_constants.OUTPUT_PATH)
    tracking_end_time = time.time()
    logger.debug('video stabilization has been finished. Running time : ' +
                 str(tracking_end_time-tracking_start_time) + ' seconds')
    time_dictionary['time_to_output'] = tracking_end_time - start_time
    project_utils.write_to_json_file(project_constants.LOGGER_NAME, time_dictionary)


if __name__ == "__main__":
    main()
