import cv2
import numpy as np
import project_constants

class image_matting:

    def __init__(self):
        self.all_frames_binary_mask = cv2.VideoCapture(project_constants.BINARY_PATH)
        self.

def generate_trimap(binary_mask, eroision_iter=6, dilate_iter=8):
    binary_mask[binary_mask == 1] = 255
    d_kernel = np.ones((3, 3))
    erode = cv2.erode(binary_mask, d_kernel, iterations=eroision_iter)
    dilate = cv2.dilate(binary_mask, d_kernel, iterations=dilate_iter)
    unknown1 = cv2.bitwise_xor(erode, binary_mask)
    unknown2 = cv2.bitwise_xor(dilate, binary_mask)
    unknowns = cv2.add(unknown1, unknown2)
    unknowns[unknowns == 255] = 127
    trimap = cv2.add(binary_mask, unknowns)
    labels = trimap.copy()
    labels[trimap == 127] = 1  # unknown
    labels[trimap == 255] = 2  # foreground
    return labels
