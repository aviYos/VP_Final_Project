import numpy as np
import cv2

# IDs
ID1 = '315488171'
ID2 = '314756297'

# General
resize_factor = 2.5


# File names
GER_NAME = '../Temp/Final_Project_Logger.log'
SECOND_INPUT_VIDEO_PATH = '../Inputs/443_motorway_with_cars_1.mp4'
INPUT_VIDEO_PATH = '../Inputs/INPUT.avi'
STABILIZE_PATH = f'../Outputs/stabilize_{ID1}_{ID2}.avi'
# STABILIZE_PATH = f'../Outputs/stabilize.avi'
# EXTRACTED_PATH = f'../Outputs/extracted_{ID1}_{ID2}.avi'
# BINARY_PATH = f'../Outputs/binary_{ID1}_{ID2}.avi'
EXTRACTED_PATH = f'../Outputs/extracted.avi'
BINARY_PATH = f'../Outputs/binary.avi'
ALPHA_PATH = f'../Outputs/alpha_{ID1}_{ID2}.avi'
MATTED_PATH = f'../Outputs/matted_{ID1}_{ID2}.avi'
OUTPUT_PATH = f'../Outputs/OUTPUT_{ID1}_{ID2}.avi'
LOGGER_NAME = '../Outputs/timing.json'
TRACKING_LOGGER = '../Outputs/tracking.json'
STABILIZED_VIDEO_PATH = '../Inputs/stabilize_315488171_314756297.avi'
BACKGROUND_SCRIBLE_PATH = r'../Temp/bg_scribbles.tiff'
FOREGROUND_SCRIBLE_PATH = r'../Temp/fg_scribbles.tiff'
BACKGROUND_IMAGE_PATH = '../Inputs/background.jpg'

# Stabilization Parameters

motion = cv2.MOTION_EUCLIDEAN  # can be either MOTION_TRANSLATION, MOTION_AFFINE, MOTION_EUCLIDEAN or MOTION_HOMOGRAPHY
sigma_mat_2D = np.array([[1000, 15, 10], [15, 1000, 10]])
sigma_mat_3D = np.array([[1000, 15, 10], [15, 1000, 10], [1000, 15, 10]])
M = 10000  # number of points in Gaussian window
# define the criteria for terminating the findTransformECC function, 2nd argument is number of iterations (default 50)
# and 3rd argument is the termination epsilon (default 0.001)
criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 0.001)

# Lucas-Kanade constants
USE_LK = True  # boolean that controls whether optic flow is calculated or not
# corner detector values
MAX_CORNERS = 50
MIN_DISTANCE = 30
QUALITY_LEVEL = 0.01
K = 0.04
# pyramid filter and derivative filters
PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

INTERPOLATION_ORDER = 3  # only 2 or 3 allowed
WINDOW_SIZE_TAU = 5
MAX_ITER_TAU = 20
NUM_LEVELS_TAU = 8
SKIP_LEVEL = 3  # -1 to not skip any pyramid levels

# number of pixels to add black borders on stabilized video
START_ROWS = 25
END_ROWS = 25
START_COLS = 20
END_COLS = 0

# Background Subtraction Parameters

Background_Subtraction_Alpha = 0.05
Background_Subtraction_T = 0.9
Background_Subtraction_Theta = 2.5
Background_Subtraction_Mask_Threshold = 175
NUMBER_OF_COLOR_CHANNELS = 3
MEDIAN_FILTER_DENSE = 55
MEDIAN_FILTER_THRESHOLD = 50
SKIN_SAT_THRESHOLD_LOW = 70
SKIN_SAT_THRESHOLD_UPPER = 140
SKIN_HUE_threshold = 15
SKIN_VALUE_threshold = 179
VALUE_NOISE_THRESHOLD = 190
TRAIN_ITER = 8

# Image matting
Rho_first_frame = 3
Rho_second_frame = 30
distance_map_radius = 1.5

# Tracking parameters
