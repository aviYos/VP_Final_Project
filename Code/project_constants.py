import numpy as np
import cv2

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
LOGGER_NAME = '../Temp/general_log.json'
TIMING_LOGGER = '../Outputs/timing.json'
TRACKING_LOGGER = '../Outputs/tracking.json'

# Stabilization Parameters

motion = cv2.MOTION_EUCLIDEAN  # either cv2.MOTION_EUCLIDEAN or cv2.MOTION_HOMOGRAPHY
sigma_mat_2D = np.array([[1000, 15, 10], [15, 1000, 10]])
sigma_mat_3D = np.array([[1000, 15, 10], [15, 1000, 10], [1000, 15, 10]])

# Background Subtraction Parameters

Background_Subtraction_Alpha = 0.05
Background_Subtraction_T = 0.9
Background_Subtraction_Theta = 2.5
Background_Subtraction_Mask_Threshold = 175
