import numpy as np
import cv2
import project_constants
from tqdm import tqdm
from scipy import signal, ndimage

SMOOTHING_RADIUS = 50


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


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
    """Coverts image to a pyramid list of size num_levels.

    First, create a list with the original image in it. Then, iterate over the
    levels. In each level, convolve the PYRAMID_FILTER with the image from the
    previous level. Then, decimate the result using indexing: simply pick
    every second entry of the result.
    Hint: Use signal.convolve2d with boundary='symm' and mode='same'.

    Args:
        image: np.ndarray. Input image.
        num_levels: int. The number of blurring / decimation times.

    Returns:
        pyramid: list. A list of np.ndarray of images.

    Note that the list length should be num_levels + 1 as the in first entry of
    the pyramid is the original image.
    You are not allowed to use cv2 PyrDown here (or any other cv2 method).
    We use a slightly different decimation process from this function.
    """
    pyramid = [image.copy()]
    for i in range(num_levels):
        previous_level_image = pyramid[i]
        """  convolve the previous_level_image with PYRAMID_FILTER """
        current_level_image = signal.convolve2d(previous_level_image, project_constants.PYRAMID_FILTER,
                                                mode='same', boundary='symm')
        """ decimation by 2 """
        current_level_image = current_level_image[::2, ::2]
        pyramid.append(current_level_image)
    return pyramid


def lucas_kanade_step(I1: np.ndarray,
                      I2: np.ndarray,
                      window_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Perform one Lucas-Kanade Step.

    This method receives two images as inputs and a window_size. It
    calculates the per-pixel shift in the x-axis and y-axis. That is,
    it outputs two maps of the shape of the input images. The first map
    encodes the per-pixel optical flow parameters in the x-axis and the
    second in the y-axis.

    (1) Calculate Ix and Iy by convolving I2 with the appropriate filters (
    see the constants in the head of this file).
    (2) Calculate It from I1 and I2.
    (3) Calculate du and dv for each pixel:
      (3.1) Start from all-zeros du and dv (each one) of size I1.shape.
      (3.2) Loop over all pixels in the image (you can ignore boundary pixels up
      to ~window_size/2 pixels in each side of the image [top, bottom,
      left and right]).
      (3.3) For every pixel, pretend the pixel’s neighbors have the same (u,
      v). This means that for NxN window, we have N^2 equations per pixel.
      (3.4) Solve for (u, v) using Least-Squares solution. When the solution
      does not converge, keep this pixel's (u, v) as zero.
    For detailed Equations reference look at slides 4 & 5 in:
    http://www.cse.psu.edu/~rtc12/CSE486/lecture30.pdf

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one is of the shape of the
        original image. dv encodes the optical flow parameters in rows and du
        in columns.
    """

    Ix = signal.convolve2d(I2, project_constants.X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, project_constants.Y_DERIVATIVE_FILTER, mode='same')
    It = I2 - I1

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    h, w = I1.shape

    """ ignore window size / 2 pixels"""
    border_index = int(window_size / 2)

    for i in range(border_index, h - border_index):
        for j in range(border_index, w - border_index):
            Ix_vec = Ix[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()
            Iy_vec = Iy[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()
            It_vec = It[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()

            A = np.vstack((Ix_vec, Iy_vec)).T
            b = -1 * It_vec

            """ Compute du and dv by : [du,dv ] = (At * A)^-1 * At * b, and skip if we have A with zero determinant"""
            try:
                out_vector = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
                du[i, j] = out_vector[0]
                dv[i, j] = out_vector[1]

            except np.linalg.LinAlgError:
                continue

    return du, dv


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Warp image using the optical flow parameters in u and v.

    Note that this method needs to support the case where u and v shapes do
    not share the same shape as of the image. We will update u and v to the
    shape of the image. The way to do it, is to:
    (1) cv2.resize to resize the u and v to the shape of the image.
    (2) Then, normalize the shift values according to a factor. This factor
    is the ratio between the image dimension and the shift matrix (u or v)
    dimension (the factor for u should take into account the number of columns
    in u and the factor for v should take into account the number of rows in v).

    As for the warping, we'll use a different method than scipy's gridddata, which was shown to be much slower.
    We use cv2's fast implementation of warpPerspective, that performs a perspective transformation.
    Finally, fill the nan holes with the source image values.

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """

    h, w = image.shape

    u_normalization_factor = h / u.shape[0]
    v_normalization_factor = w / v.shape[0]

    resized_u = cv2.resize(u, image.T.shape) * u_normalization_factor
    resized_v = cv2.resize(v, image.T.shape) * v_normalization_factor

    x_range = np.linspace(0, w - 1, w)
    y_range = np.linspace(0, h - 1, h)

    x_grid, y_grid = np.meshgrid(x_range, y_range)

    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    grid_x_plus_du = x_grid + resized_u.flatten()
    grid_y_plus_dv = y_grid + resized_v.flatten()

    cord = [grid_x_plus_du, grid_y_plus_dv]
    interpolated_image = ndimage.map_coordinates(image.T, coordinates=cord, order=project_constants.INTERPOLATION_ORDER,
                                                 cval=np.nan)

    """ resize interpolated image  """

    interpolated_image_original_size = interpolated_image.reshape(image.shape)

    """ store original values instead of nan values  """

    interpolated_image_original_size[np.isnan(interpolated_image_original_size)] = \
        image[np.isnan(interpolated_image_original_size)]

    image_warp = interpolated_image_original_size

    return image_warp


def movingAverage(curve, radius):
    window_size = 2 * radius + 1
    # Define the filter
    f = np.ones(window_size) / window_size
    # Add padding to the boundaries
    curve_pad = np.lib.pad(curve, (radius, radius), 'edge')
    # Apply convolution
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    # Remove padding
    curve_smoothed = curve_smoothed[radius:-radius]
    # return smoothed curve
    return curve_smoothed


def smooth(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    # Filter the x, y and angle curves
    for i in range(3):
        smoothed_trajectory[:, i] = movingAverage(trajectory[:, i], radius=SMOOTHING_RADIUS)

    return smoothed_trajectory


def fixBorder(frame):
    s = frame.shape
    # Scale the image 4% without moving the center
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    frame = cv2.warpAffine(frame, T, (s[1], s[0]))
    return frame


def find_image_corner_pixels(I2: np.ndarray, window_size: int, k: float) \
        -> list[tuple[np.ndarray, np.ndarray]]:
    border_size = int(window_size / 2)
    h, w = I2.shape
    edge_mask = np.ones((h, w), dtype=np.uint8)
    top_left = (h - border_size, border_size)
    bottom_right = (border_size, w - border_size)
    cv2.rectangle(edge_mask, top_left, bottom_right, 0, cv2.FILLED)
    corner_pixels = cv2.goodFeaturesToTrack(np.float32(I2), mask=edge_mask, maxCorners=project_constants.MAX_CORNERS,
                                            blockSize=window_size, qualityLevel=project_constants.QUALITY_LEVEL, k=k,
                                            minDistance=project_constants.MIN_DISTANCE)
    corners = []
    for corner_group in corner_pixels:
        corners.append((corner_group[0][0], corner_group[0][1]))
    return corners


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int, k: float) -> tuple[np.ndarray, np.ndarray]:
    """Faster implementation of a single Lucas-Kanade Step.

    (1) If the image is small enough (you need to design what is good
    enough), simply return the result of the good old lucas_kanade_step
    function.
    (2) Otherwise, find corners in I2 and calculate u and v only for these
    pixels.
    (3) Return maps of u and v which are all zeros except for the corner
    pixels you found in (2).

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """

    h, w = I1.shape

    # if image size smaller than 50 x 50 - use regular LK step
    if h <= project_constants.SMALL_ENOUGH_H and w <= project_constants.SMALL_ENOUGH_W:
        return lucas_kanade_step(I1, I2, window_size)

    # find corners in I2

    I2_corners = find_image_corner_pixels(I2, window_size, k=k)

    Ix = signal.convolve2d(I2, project_constants.X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, project_constants.Y_DERIVATIVE_FILTER, mode='same')
    It = I2 - I1

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)
    """ ignore window size / 2 pixels"""
    border_index = int(window_size / 2)

    for corner_pixel in I2_corners:

        i, j = int(corner_pixel[1]), int(corner_pixel[0])

        Ix_vec = Ix[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()
        Iy_vec = Iy[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()
        It_vec = It[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()

        A = np.vstack((Ix_vec, Iy_vec)).T
        b = -1 * It_vec

        """ Compute du and dv by : [du,dv ] = (At * A)^-1 * At * b, and skip if we have A with zero determinant"""
        try:
            out_vector = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
            du[i, j] = out_vector[0]
            dv[i, j] = out_vector[1]

        except np.linalg.LinAlgError:
            continue

    return du, dv


def faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels .

    Use faster_lucas_kanade_step instead of lucas_kanade_step.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the shift in rows and u in columns.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    u_size = (IMAGE_SIZE[1], IMAGE_SIZE[0])
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    pyramid_I1 = build_pyramid(I1, num_levels)  # create levels list for I1
    pyramid_I2 = build_pyramid(I2, num_levels)  # create levels list for I1

    # start from u and v in the size of smallest image
    u = np.zeros(pyramid_I2[-1].shape)
    v = np.zeros(pyramid_I2[-1].shape)

    for level_index in range(num_levels, -1, -1):
        if project_constants.SKIP_LEVEL == level_index:
            break
        warped_I2 = warp_image(pyramid_I2[level_index], u, v)
        # calculate parameters that don't change in each iteration

        for iter_idx in range(max_iter):
            du, dv = faster_lucas_kanade_step(pyramid_I1[level_index], warped_I2, window_size, project_constants.K)
            u += du
            v += dv
            warped_I2 = warp_image(pyramid_I2[level_index], u, v)

        if level_index:
            u = 2 * cv2.resize(u, (pyramid_I1[level_index - 1].shape[1], pyramid_I1[level_index - 1].shape[0]))
            v = 2 * cv2.resize(v, (pyramid_I1[level_index - 1].shape[1], pyramid_I1[level_index - 1].shape[0]))
    if u.shape != u_size:
        u = cv2.resize(u, IMAGE_SIZE)
    if v.shape != u_size:
        v = cv2.resize(v, IMAGE_SIZE)
    return u, v


def stabilize_video_new(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    cap = cv2.VideoCapture(input_video_path)
    # in order to use tqdm to monitor the progress, we require the number of frames in the video
    num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=num_of_frames)
    vid_params = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = vid_params.get("fps")
    size = vid_params.get("width"), vid_params.get("height")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    isFirstFrame = True
    border_index = int(window_size / 2)

    for i in range(num_of_frames - 2):
        ret, frame = cap.read()
        if not ret:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if isFirstFrame:
            h_factor = int(np.ceil(gray_frame.shape[0] / (2 ** (num_levels - 1))))
            w_factor = int(np.ceil(gray_frame.shape[1] / (2 ** (num_levels - 1))))
            IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                          h_factor * (2 ** (num_levels - 1)))

            u = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
            v = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
            prev_u = u
            prev_v = v

            prev_frame = gray_frame
            transforms = np.zeros((num_of_frames - 1, 3), np.float32)
            isFirstFrame = False
        else:
            if gray_frame.shape != size:
                gray_frame = cv2.resize(gray_frame, size)
            (new_u, new_v) = faster_lucas_kanade_optical_flow(prev_frame, gray_frame, window_size, max_iter, num_levels)
            h, w = new_u.shape
            avg_u_corners = np.nanmean(np.where(new_u == 0, np.nan, new_u))
            avg_v_corners = np.nanmean(np.where(new_v == 0, np.nan, new_v))
            if np.isnan(avg_u_corners):
                avg_u_corners = 0
            if np.isnan(avg_v_corners):
                avg_v_corners = 0
            u_avg = avg_u_corners * np.ones(new_u.shape)
            v_avg = avg_v_corners * np.ones(new_v.shape)
            u = new_u
            v = new_v
            u[border_index:h - border_index, border_index:w - border_index] = u_avg[border_index:h - border_index,
                                                                              border_index:w - border_index] + prev_u[
                                                                                                               border_index:h - border_index,
                                                                                                               border_index:w - border_index]
            v[border_index:h - border_index, border_index:w - border_index] = v_avg[border_index:h - border_index,
                                                                              border_index:w - border_index] + prev_v[
                                                                                                               border_index:h - border_index,
                                                                                                               border_index:w - border_index]

            prev_corners = find_image_corner_pixels(prev_frame, window_size, project_constants.K)
            this_corners = []
            src_corners = []
            for corner in prev_corners:
                x, y = int(corner[1]), int(corner[0])
                if (x > (prev_frame.shape[0] - border_index)) or (y > (prev_frame.shape[1] - border_index)):
                    continue
                if x < border_index or y < border_index:
                    continue
                src_corners.append([x, y])
                this_corners.append([int(x + u[x, y]), int(y + v[x, y])])
            src_corners = np.array(src_corners)
            this_corners = np.array(this_corners)
            m, _ = cv2.estimateAffinePartial2D(src_corners, this_corners)
            dx = m[0, 2]
            dy = m[1, 2]
            # Extract rotation angle
            da = np.arctan2(m[1, 0], m[0, 0])
            transforms[i] = [dx, dy, da]

            prev_u = u
            prev_v = v
            prev_frame = gray_frame
            pbar.update(1)

    trajectory = np.cumsum(transforms, axis=0)

    smoothed_trajectory = smooth(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for i in range(num_of_frames - 2):
        ret, frame = cap.read()
        if not ret:
            break

        # Extract transformations from the new transformation array
        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        # Reconstruct transformation matrix accordingly to new values
        m = np.zeros((2, 3), np.float32)
        m[0, 0] = np.cos(da)
        m[0, 1] = -np.sin(da)
        m[1, 0] = np.sin(da)
        m[1, 1] = np.cos(da)
        m[0, 2] = dx
        m[1, 2] = dy

        # Apply affine wrapping to the given frame
        frame_stabilized = cv2.warpAffine(frame, m, size)

        # Fix border artifacts
        frame_stabilized = fixBorder(frame_stabilized)

        # If the image is too big, resize it.
        if frame_stabilized.shape != size:
            frame_stabilized = cv2.resize(frame_stabilized, size)
        out.write(np.uint8(frame_stabilized))
        if cv2.waitKey(1) == ord('q'):
            break

    pbar.close()
    cv2.destroyAllWindows()
    cap.release()
    out.release()
