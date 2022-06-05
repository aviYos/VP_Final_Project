import project_constants
import time
from feature_point_gaussian_stabilization import stabilize_video_with_gaussian
from background_subtraction import background_subtractor
from matting import matting
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
    stabilize_video_with_gaussian(project_constants.INPUT_VIDEO_PATH, project_constants.STABILIZE_PATH)
    stabilization_end_time = time.time()
    stabilization_time = stabilization_end_time - start_time
    logger.debug('video stabilization has finished. Runtime: ' +
                 str(stabilization_time) + ' seconds')
    time_dictionary['time_to_stabilize'] = stabilization_time

    # Background Subtraction
    logger.debug("Running background subtraction. Stabilized path: " + project_constants.STABILIZE_PATH)
    print('Subtracting background:')
    background_start_time = time.time()
    background_subtractor_handle = background_subtractor()
    background_subtractor_handle.main_background_subtraction_module()
    background_end_time = time.time()
    background_time = background_end_time - start_time
    logger.debug('background subtraction has finished. Runtime: ' +
                 str(background_end_time-background_start_time) + ' seconds')
    time_dictionary['time_to_binary'] = background_time

    # Matting
    logger.debug("Running matting. Binary path: " + project_constants.BINARY_PATH + ". Extracted path:" +
                 project_constants.EXTRACTED_PATH)
    print('Matting extracted video to background image:')
    matting_start_time = time.time()
    mat = matting()
    mat.main_image_matting_module()
    matting_end_time = time.time()
    matting_time = matting_end_time - start_time
    logger.debug('matting has finished. Runtime: ' +
                 str(matting_end_time - matting_start_time) + ' seconds')
    time_dictionary['time_to_alpha'] = matting_time
    time_dictionary['time_to_matted'] = matting_time

    # Tracking
    logger.debug("Running tracking program. Matted path: " + project_constants.MATTED_PATH)
    tracking_start_time = time.time()
    tracking(project_constants.MATTED_PATH, project_constants.BINARY_PATH, project_constants.OUTPUT_PATH)
    tracking_end_time = time.time()
    logger.debug('Tracking has finished. Runtime: ' +
                 str(tracking_end_time-tracking_start_time) + ' seconds')
    time_dictionary['time_to_output'] = tracking_end_time - start_time
    project_utils.write_to_json_file(project_constants.LOGGER_NAME, time_dictionary)


if __name__ == "__main__":
    main()
