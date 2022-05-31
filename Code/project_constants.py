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
TRACKING_LOGGER = '../Outputs/tracking.json'
STABILIZED_VIDEO_PATH = '../Inputs/stabilize_315488171_314756297.avi'
BACKGROUND_SCRIBLE_PATH = r'../Temp/bg_scribbles.tiff'
FOREGROUND_SCRIBLE_PATH = r'../Temp/fg_scribbles.tiff'
BACKGROUND_IMAGE_PATH = '../Inputs/background.jpg'

# Stabilization Parameters

motion = cv2.MOTION_HOMOGRAPHY  # can be either MOTION_TRANSLATION, MOTION_AFFINE, MOTION_EUCLIDEAN or MOTION_HOMOGRAPHY
# sigma matrices for gaussian smoothing
sigma_mat_2D = np.array([[1000, 15, 10], [15, 1000, 10]])
sigma_mat_3D = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
M = 10000  # number of points in Gaussian window

# number of pixels to add black borders on stabilized video
START_ROWS = 70
END_ROWS = 30
START_COLS = 25
END_COLS = 25
INDEX_TO_ADD_CROP = 15
START_COLS_NEXT = 200

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
distance_map_radius = 6
