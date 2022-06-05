import numpy as np
import cv2

# IDs
ID1 = '315488171'
ID2 = '314756297'

# General
resize_factor = 1


# File names
GER_NAME = '../Temp/Final_Project_Logger.log'
INPUT_VIDEO_PATH = '../Inputs/INPUT.mp4'
#STABILIZE_PATH = f'../Outputs/stabilize_{ID1}_{ID2}.avi'
#EXTRACTED_PATH = f'../Outputs/extracted_{ID1}_{ID2}.mp4'
#BINARY_PATH = f'../Outputs/binary_{ID1}_{ID2}.avi'
ALPHA_PATH = f'../Outputs/alpha_{ID1}_{ID2}.avi'
MATTED_PATH = f'../Outputs/matted_{ID1}_{ID2}.avi'
OUTPUT_PATH = f'../Outputs/OUTPUT_{ID1}_{ID2}.avi'
LOGGER_NAME = '../Outputs/timing.json'
TRACKING_LOGGER = '../Outputs/tracking.json'
STABILIZED_VIDEO_PATH = '../Inputs/stabilize_315488171_314756297.avi'
BACKGROUND_IMAGE_PATH = '../Inputs/background.jpg'
STABILIZE_PATH = f'../Outputs/stabilize.avi'
BINARY_PATH = f'../Outputs/binary.avi'
EXTRACTED_PATH = f'../Outputs/extracted.avi'

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
END_COLS = 75
INDEX_TO_ADD_CROP = -1
CROP_FROM_START = 1
START_COLS_NEXT = 75

# Background Subtraction Parameters
TRAIN_ITER = 8

# Image matting
distance_map_radius = 0.3
