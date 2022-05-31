import numpy as np
import cv2
from project_utils import load_entire_video
from tqdm import tqdm
import project_constants
from scipy import signal, ndimage


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
# descriptors from the image (not unlike SIFT or Harris Corner Detector)
def detector(image1, image2):
    sift = cv2.xfeatures2d.SIFT_create()

    keypoints_1, descriptors_1 = sift.detectAndCompute(image1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(image2, None)
    return keypoints_1, descriptors_1, keypoints_2, descriptors_2


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'reflect')

    '''Fix padding manually'''
    for i in range(radius):
        curve_pad[i] = curve_pad[radius] - curve_pad[i]

    for i in range(len(curve_pad) - 1, len(curve_pad) - 1 - radius, -1):
        curve_pad[i] = curve_pad[len(curve_pad) - radius - 1] - curve_pad[i]

    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    return curve_smoothed


def smooth(trajectory, smooth_radius):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(smoothed_trajectory.shape[1]):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=smooth_radius)
    return smoothed_trajectory


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


def stabilize_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    vid_params = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = vid_params.get("fps")
    size = vid_params.get("width"), vid_params.get("height")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    frames_bgr = load_entire_video(cap, color_space='bgr')
    w = size[0]
    h = size[1]
    num_frames = len(frames_bgr)
    prev = frames_bgr[0]
    prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = np.zeros((num_frames-1, 3), np.float32)
    print('Finding warp matrices:')
    pbar = tqdm(total=num_frames-1)
    for frame_index, curr in enumerate(frames_bgr[1:]):
        # prev_pts = cv2.goodFeaturesToTrack(prev_gray,
        #                                   maxCorners=project_constants.MAX_CORNERS,
        #                                   qualityLevel=project_constants.QUALITY_LEVEL,
        #                                   minDistance=project_constants.MIN_DISTANCE,
        #                                   blockSize=project_constants.BLOCK_SIZE)

        curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        key_pt1, descrip1, key_pt2, descrip2 = detector(prev_gray, curr_gray)

        brute_force = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
        matches = brute_force.match(descrip1, descrip2)
        # finding the humming distance of the matches and sorting them
        matches = sorted(matches, key=lambda x: x.distance)
        prev_pts = np.array([key_pt1[mat.queryIdx].pt for mat in matches])
        curr_pts = np.array([key_pt2[mat.trainIdx].pt for mat in matches])
        # Find transformation matrix
        m, _ = cv2.estimateAffinePartial2D(prev_pts, curr_pts)
        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        transforms[frame_index] = [dx, dy, da]
        prev_gray = curr_gray
        pbar.update(1)
    pbar.close()
    # Compute trajectory using cumulative sum of transformations
    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = smooth(trajectory, project_constants.SMOOTH_RADIUS)
    # Calculate difference in smoothed_trajectory and trajectory
    difference = smoothed_trajectory - trajectory

    # Calculate smooth transformation array
    transforms_smooth = transforms + difference

    # Write n_frames-1 transformed frames
    print('Applying warps to frame:')
    pbar = tqdm(total=num_frames-1)
    for frame_index, frame in enumerate(frames_bgr[:-1]):
        # Apply affine wrapping to the given frame

        # Extract transformations from the new transformation array
        dx = transforms_smooth[frame_index, 0]
        dy = transforms_smooth[frame_index, 1]
        da = transforms_smooth[frame_index, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        new_frame = cv2.warpAffine(frame, m, size)
        # new_frame = cv2.warpPerspective(frame, transform_matrix, (w, h))
        new_frame = add_borders(new_frame, project_constants.START_ROWS, project_constants.END_ROWS,
                                project_constants.START_COLS, project_constants.END_COLS)
        out.write(np.uint8(new_frame))
        pbar.update(1)
    pbar.close()
    cv2.destroyAllWindows()
    cap.release()
    out.release()
    print('~~~~~~~~~~~ [Video Stabilization] FINISHED! ~~~~~~~~~~~')