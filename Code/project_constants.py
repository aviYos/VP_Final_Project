import numpy as np

# IDs
ID1 = '315488171'
ID2 = '314756297'

# General
resize_factor = 2.5


# File names
GER_NAME = '../Temp/Final_Project_Logger.log'
SECOND_INPUT_VIDEO_PATH = '../Inputs/443_motorway_with_cars_1.mp4'
INPUT_VIDEO_PATH = '../Inputs/INPUT.mp4'
#STABILIZE_PATH = f'../Outputs/stabilize_{ID1}_{ID2}.avi'
STABILIZE_PATH = f'../Outputs/stabilize.avi'
#EXTRACTED_PATH = f'../Outputs/extracted_{ID1}_{ID2}.mp4'
#BINARY_PATH = f'../Outputs/binary_{ID1}_{ID2}.avi'
EXTRACTED_PATH = f'../Outputs/extracted.avi'
BINARY_PATH = f'../Outputs/binary.avi'
ALPHA_PATH = f'../Outputs/alpha_{ID1}_{ID2}.avi'
MATTED_PATH = f'../Outputs/matted_{ID1}_{ID2}.avi'
OUTPUT_PATH = f'../Outputs/OUTPUT_{ID1}_{ID2}.avi'
LOGGER_NAME = '../Outputs/timing.json'
TRACKING_LOOGER = '../Outputs/tracking.json'
STABILIZED_VIDEO_PATH = '../Inputs/stabilize_315488171_314756297.avi'
BACKGROUND_SCRIBLE_PATH = r'../Temp/bg_scribbles.tiff'
FOREGROUND_SCRIBLE_PATH = r'../Temp/fg_scribbles.tiff'
BACKGROUND_IMAGE_PATH = '../Inputs/background.jpg'

# Corner Detector parameters
MAX_CORNERS = 40
MIN_DISTANCE = 2
QUALITY_LEVEL = 0.1
K = 0.04  # previously 0.137

# Lucas Kanade Video Stabilization parameters
PYRAMID_FILTER = 1.0 / 256 * np.array([[1, 4, 6, 4, 1],
                                       [4, 16, 24, 16, 4],
                                       [6, 24, 36, 24, 6],
                                       [4, 16, 24, 16, 4],
                                       [1, 4, 6, 4, 1]])
X_DERIVATIVE_FILTER = np.array([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]])
Y_DERIVATIVE_FILTER = X_DERIVATIVE_FILTER.copy().transpose()

SMALL_ENOUGH_H = 50
SMALL_ENOUGH_W = 50

WINDOW_SIZE_TAU = 5
MAX_ITER_TAU = 5
NUM_LEVELS_TAU = 5

START_ROWS = 15
START_COLS = 5
END_ROWS = 10
END_COLS = 35

# Backgound Subtraction Parameters

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