import numpy as np
import cv2
from project_utils import load_entire_video
from tqdm import tqdm
import project_constants
from scipy import signal, ndimage


# function that extracts input video parameters
def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.
    Args:
        capture: cv2.VideoCapture object.
    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


# function to detect the features by finding key points and
# descriptors from the image using SIFT
def detector(image1, image2):
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)
    return keypoints_1, descriptors_1, keypoints_2, descriptors_2


# function that calculates the smoothed trajectory using Gaussian filters.
# if sigma is 0 then the kernel is the zero matrix - meaning that there is no smoothing.
def gauss_convolve(trajectory, window, sigma):
    if sigma != 0:
        kernel = signal.gaussian(window, std=sigma)
        kernel = kernel / np.sum(kernel)
    else:
        kernel = np.zeros(window)
    return ndimage.convolve(trajectory, kernel, mode='reflect')


# function that calculates the camera trajectory, and the smoothed trajectory by convolving with a gaussian kernel
# or zero matrix, and the smoothed warp stack.
def moving_average(warp_stack, sigma_mat):
    x, y = warp_stack.shape[1:]
    original_trajectory = np.cumsum(warp_stack, axis=0)
    smoothed_trajectory = np.zeros(original_trajectory.shape)
    for i in range(x):
        for j in range(y):
            smoothed_trajectory[:, i, j] = gauss_convolve(original_trajectory[:, i, j], project_constants.M,
                                                          sigma_mat[i, j])
    smoothed_warp = np.apply_along_axis(lambda m:
                                        ndimage.convolve(m, [0, 1, -1], mode='reflect'),
                                        axis=0, arr=smoothed_trajectory)
    return smoothed_warp, smoothed_trajectory, original_trajectory


# function that adds black borders to the stabilized video to reduce noise due to moving borders
def add_borders(frame: np.ndarray, start_rows: int, end_rows: int, start_cols: int, end_cols: int) -> np.ndarray:
    h, w, _ = frame.shape
    start_rows = min(h, start_rows)
    start_cols = min(w, start_cols)
    end_rows = min(h, end_rows)
    end_cols = min(w, end_cols)
    frame[:start_rows, :] = 0
    frame[h-end_rows:, :] = 0
    frame[:, :start_cols] = 0
    frame[:, w-end_cols:] = 0
    return frame


# function that calculates the homography matrix H_tot by multiplying warps defined in warp_stack.
# if the motion is affine, then create a 3x3 matrix from each 2x3 warp matrix.
def homography_gen(warp_stack):
    H_tot = np.eye(3)
    if project_constants.motion == cv2.MOTION_HOMOGRAPHY:
        wsp = warp_stack
    else:
        wsp = np.dstack([warp_stack[:, 0, :], warp_stack[:, 1, :], np.array([[0, 0, 1]]*warp_stack.shape[0])])
    for i in range(len(warp_stack)):
        if project_constants.motion == cv2.MOTION_HOMOGRAPHY:
            H_tot = np.matmul(wsp[i], H_tot)
        else:
            H_tot = np.matmul(wsp[i].T, H_tot)
        yield np.linalg.inv(H_tot)


# function that calculates the maximum deviation at each edge of the frame from the corners of the images due to warping
def get_border_pads(img_shape, warp_stack) -> (int, int, int, int):
    top = 0
    bottom = 0
    left = 0
    right = 0
    w, h = img_shape[0], img_shape[1]
    tr_point = np.array([w, h, 1])
    bl_point = np.array([0, 0, 1])
    for i in range(len(warp_stack)):
        tr = np.matmul(warp_stack[i], tr_point)
        bl = np.matmul(warp_stack[i], bl_point)
        left = max(left, np.abs(min(bl[0], 0)))
        bottom = max(bottom, np.abs(min(bl[1], 0)))
        right = max(right, np.abs(max(0, tr[0]-w)))
        top = max(top, np.abs(max(0, tr[1]-h)))
    return int(top), int(bottom), int(left), int(right)


# function that updates the homography matrix with border pads from get_border_pads and warps each
# image with H_tot within the window size defined by the window shape and pads to get the stabilized image stack.
def apply_warping_fullview(images, warp_stack, num_of_frames):
    top, bottom, left, right = get_border_pads(img_shape=images[0].shape, warp_stack=warp_stack)
    H = homography_gen(warp_stack)
    imgs = []
    pbar = tqdm(total=num_of_frames-1)
    for i, img in enumerate(images[1:]):
        H_tot = next(H)+np.array([[0, 0, left], [0, 0, top], [0, 0, 0]])
        img_warp = cv2.warpPerspective(img, H_tot, (img.shape[1]+left+right, img.shape[0]+top+bottom))
        imgs += [img_warp]
        pbar.update(1)
    pbar.close()
    return imgs


# main stabilization function that performs:
# 1) finds and matches between feature points and their descriptors, using SIFT and BFMatcher
# 2) creates the warp matrix between the previous and current points using RANSAC
# 3) performs gaussian smoothing or no smoothing on the image
# 4) warps each frame according to the homography matrix created by multiplying the previous warp matrices
# 5) adds black borders to the stabilized frames and places these frames in output video
def stabilize_video_with_gaussian(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    vid_params = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = vid_params.get("fps")
    size = vid_params.get("width"), vid_params.get("height")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    frames_bgr = load_entire_video(cap, color_space='bgr')
    num_frames = vid_params.get("frame_count")
    prev = frames_bgr[0]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    print('Finding warp matrices:')
    pbar = tqdm(total=num_frames-1)
    warp_stack = []
    for frame_index, curr in enumerate(frames_bgr[1:]):
        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        key_pt1, descrip1, key_pt2, descrip2 = detector(prev_gray, curr_gray)

        # finding the L1 distance of the matches and sort them
        brute_force = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = brute_force.match(descrip1, descrip2)
        matches = sorted(matches, key=lambda x: x.distance)
        # create list of previous points and matched current points
        prev_pts = np.array([key_pt1[mat.queryIdx].pt for mat in matches])
        curr_pts = np.array([key_pt2[mat.trainIdx].pt for mat in matches])
        # Find transformation matrix
        if project_constants.motion == cv2.MOTION_HOMOGRAPHY:
            m, _ = cv2.findHomography(prev_pts, curr_pts, cv2.RANSAC, 5.0)
        else:
            m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        warp_stack += [m]
        prev_gray = curr_gray
        pbar.update(1)
    warp_stack = np.array(warp_stack)
    pbar.close()

    # Write n_frames-1 transformed frames
    print('Applying warps to frame:')
    # Perform Gaussian smoothing or no smoothing according to sigma_mat
    if project_constants.motion == cv2.MOTION_HOMOGRAPHY:
        sigma_mat = project_constants.sigma_mat_3D
    else:
        sigma_mat = project_constants.sigma_mat_2D
    smoothed_warp, smoothed_trajectory, original_trajectory = moving_average(warp_stack,
                                                                             sigma_mat=sigma_mat)
    # Warp using the homography matrix with the warp stack being the difference between the original and smooth warps
    new_images = apply_warping_fullview(images=frames_bgr, warp_stack=warp_stack - smoothed_warp,
                                        num_of_frames=num_frames)
    # Print to output video each frame and add borders according to default parameters or max top left corner
    first_frame = frames_bgr[0]
    left = project_constants.START_COLS
    if project_constants.CROP_FROM_START == 1:
        left = project_constants.START_COLS_NEXT
    first_frame = add_borders(first_frame, project_constants.START_ROWS, project_constants.END_ROWS,
                              left, project_constants.END_COLS)
    out.write(np.uint8(first_frame))
    left = project_constants.START_COLS
    for i in range(num_frames-1):
        new_frame = new_images[i]
        if new_frame.size != size:
            new_frame = cv2.resize(new_frame, size)
        if i >= project_constants.INDEX_TO_ADD_CROP:
            left = project_constants.START_COLS_NEXT
        new_frame = add_borders(new_frame, project_constants.START_ROWS, project_constants.END_ROWS,
                                left, project_constants.END_COLS)
        out.write(np.uint8(new_frame))
        pbar.update(1)
    pbar.close()
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    print('~~~~~~~~~~~ [Video Stabilization] FINISHED! ~~~~~~~~~~~')
