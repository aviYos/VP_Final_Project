import numpy as np

# IDs
ID1 = '315488171'
ID2 = '314756297'

# File names
GER_NAME = '../Temp/Final_Project_Logger.log'
INPUT_VIDEO_PATH = '../Inputs/INPUT.avi'
STABILIZE_PATH = f'../Outputs/stabilize_{ID1}_{ID2}.avi'
EXTRACTED_PATH = f'../Outputs/extracted_{ID1}_{ID2}.avi'
BINARY_PATH = f'../Outputs/binary_{ID1}_{ID2}.avi'
ALPHA_PATH = f'../Outputs/alpha_{ID1}_{ID2}.avi'
MATTED_PATH = f'../Outputs/matted_{ID1}_{ID2}.avi'
OUTPUT_PATH = f'../Outputs/OUTPUT_{ID1}_{ID2}.avi'
LOGGER_NAME = '../Outputs/timing.json'
TRACKING_LOOGER = '../Outputs/tracking.json'

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

Background_Subtraction_Alpha = 0.9
Background_Subtraction_T = 0.8
