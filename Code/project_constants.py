import numpy as np

# IDs
ID1 = '315488171'
ID2 = '314756297'

# File names
GER_NAME = '../Temp/Final_Project_Logger.log'
INPUT_VIDEO_PATH = '../Inputs/INPUT.mp4'
STABILIZE_PATH = f'../Outputs/stabilize_{ID1}_{ID2}.avi'
EXTRACTED_PATH = f'../Outputs/extracted_{ID1}_{ID2}.avi'
BINARY_PATH = f'../Outputs/binary_{ID1}_{ID2}.avi'
ALPHA_PATH = f'../Outputs/alpha_{ID1}_{ID2}.avi'
MATTED_PATH = f'../Outputs/matted_{ID1}_{ID2}.avi'
OUTPUT_PATH = f'../Outputs/OUTPUT_{ID1}_{ID2}.avi'
LOGGER_NAME = '../Outputs/timing.json'
TRACKING_LOOGER = '../Outputs/tracking.json'

# Harris Corner Detector parameters
TILES_NUM_ROW = 2
TILES_NUM_COL = 5
THRESHOLD = 8.3e9
APERTURE_SIZE = 3
K = 0.137

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