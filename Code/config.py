import cv2
import numpy as np

# generalconstants #
resize_factor = 2.5

path_to_LogFile = '..\\Outputs\\RunTimeLog.txt'  # running from CODE\ directory

path_to_shaky_input = '..\\Input\\INPUT.avi'  # running from CODE\ directory

path_to_stabilized = '..\\Outputs\\stabilized.avi'  # running from CODE\ directory

path_to_transformations_list = '..\\Temp\\transforms_file.npy'  # running from CODE\ directory

path_to_binary = '..\\Outputs\\binary.avi'  # running from CODE\ directory

path_to_extracted = '..\\Outputs\\extracted.avi'  # running from CODE\ directory

path_to_matted = '..\\Outputs\\matted.mp4'  # running from CODE\ directory

path_to_alpha = '..\\Outputs\\alpha.mp4'  # running from CODE\ directory

path_to_unstabilized_alpha = '..\\Outputs\\unstabilized_alpha.avi'  # running from CODE\ directory

path_to_OUTPUT = '..\\Outputs\\OUTPUT.avi'  # running from CODE\ directory

path_to_FG_scribbles = '..\\Temp\\fg_scribbles.tiff'  # running from CODE\ directory

path_to_BG_scribbles = '..\\Temp\\bg_scribbles.tiff'  # running from CODE\ directory

path_to_new_BG = '..\\Input\\background.jpg'  # running from CODE\ directory
# generalconstants #


# stabilization constants #
radius = 50  # moving median window radius
num_of_trnsf_prm = 3  # number of parameters in affine transformation [dx,dy,dtheta]
fixBorder_scale = 1.04  # fixBorder scale size
maxCorners = 200
qualityLevel = 0.01
minDistance = 130
blockSize = 3
winSize = (11, 11)
# stabilization constants #

# background subtraction constants #
KNN_history = 10
KNN_dist2Threshold = 300
HSV_lower_bound = [0, 100, 0]
HSV_upper_bound = [50, 255, 183]
temporal_median_depth = 55
temporal_median_threshold = 50
skin_H_threshold = 15
skin_S_threshold_lower = 70
skin_S_threshold_upper = 140
skin_V_threshold = 179
pants_H_threshold = 90
pants_S_threshold = 110
pants_V_threshold = 40
H_gamma = 0.2
S_gamma = 2
V_gamma = 2.5
V_threshold_noise_reduction = 190
shoes_B_threshold = 100
shoes_G_threshold = 100
shoes_R_threshold = 100
# background subtraction constants #

# matting constants #
RED_value = 249  # RED value absent from histogram of the first frame in Stabilized Vid
r = 1.5 # The power of the distance map in the refinement stage
rho1 = 3 # uncertainty radius for the first frame
rho2 = 30 # uncertainty radius for the rest of the frames
# matting constants #

# tracker constants #
xmin = 17
ymin = 191
boxwidth = 459
boxheight = 795
# tracker constants #

# This function uses connected components to delete all components accept for the largest one.
# This helps with denoising backgrounf pixels.
def save_largest_blob(I):
    C = I.copy()
    num_of_labels, component_labels, stats, centroids = cv2.connectedComponentsWithStats(C)
    sizes = stats[range(num_of_labels), cv2.CC_STAT_AREA]
    max_ind = np.argmax(sizes)
    sizes[max_ind] = -1  # neutralize background label so foreground is largest
    largest_blob = np.argmax(sizes)
    C[component_labels != largest_blob] = 0
    rect_indices = np.array([stats[largest_blob, cv2.CC_STAT_LEFT],  # x - left most(col)
                             stats[largest_blob, cv2.CC_STAT_TOP],  # y - top most (row)
                             stats[largest_blob, cv2.CC_STAT_WIDTH],  # WIDTH - the col number for the image
                             stats[largest_blob, cv2.CC_STAT_HEIGHT]])  # the height number(rows) for the image

    blob_center = (int(centroids[largest_blob, 0]), int(centroids[largest_blob, 1]))
    return C, rect_indices, blob_center


def write2log(module, exec_time):
    with open(path_to_LogFile, 'a') as writer:
        if module is 'RunAll':
            string_to_write = 'Time taken to run all programs consecutively in seconds: ' + str(exec_time) + '\n'
        else:
            string_to_write = 'Time taken for ' + module + ' in seconds: ' + str(exec_time) + '\n'
        writer.write(string_to_write)


def clearLog():
    with open(path_to_LogFile, 'w') as writer:
        writer.write('')