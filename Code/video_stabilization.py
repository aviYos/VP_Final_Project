import cv2
import numpy as np
import project_constants
from tqdm import tqdm
from scipy import signal, ndimage

ID1 = 315488171
ID2 = 314756297


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
        h, w = previous_level_image.shape
        current_level_image = cv2.pyrDown(previous_level_image, dstsize=(w//2, h//2))
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


def count_frames_manual(cap):
    total = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        total += 1
        if cv2.waitKey(1) == ord('q'):
            break
    return total


def find_image_corner_pixels(I2: np.ndarray, window_size: int, k: float) \
        -> list[tuple[np.ndarray, np.ndarray]]:
    border_size = int(window_size/2)
    h, w = I2.shape
    edge_mask = np.ones((h, w), dtype=np.uint8)
    top_left = (h-border_size, border_size)
    bottom_right = (border_size, w-border_size)
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

    return u, v


def even_faster_lucas_kanade_optical_flow(
        I1: np.ndarray, I2: np.ndarray, window_size: int, max_iter: int,
        num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels without iterative warping.

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

        warped_I2 = warp_image(pyramid_I2[level_index], u, v)

        # calculate all parameters before iterative section
        I2_corners = find_image_corner_pixels(I2, window_size, k=project_constants.K)

        Ix = signal.convolve2d(warped_I2, project_constants.X_DERIVATIVE_FILTER, mode='same')
        Iy = signal.convolve2d(warped_I2, project_constants.Y_DERIVATIVE_FILTER, mode='same')
        border_index = int(window_size / 2)
        # calculate absolute gradient image
        abs_grad_x = cv2.convertScaleAbs(Ix)
        abs_grad_y = cv2.convertScaleAbs(Iy)
        grad_I = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        A_stack = []  # LK matrix for each window around corner pixel
        r_stack = []  # offset vector r
        s_stack = []  # interpolation points s
        num_corner_pixels = 0
        for corner_pixel in I2_corners:
            num_corner_pixels += 1
            i, j = int(corner_pixel[1]), int(corner_pixel[0])
            Ix_vec = Ix[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()
            Iy_vec = Iy[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()
            A_stack.append(np.vstack((Ix_vec, Iy_vec)).T)
            r_stack.append(-1 * np.mulmat(grad_I[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1],
                                          warped_I2[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1]))
            s = np.zeros(project_constants.INTERPOLATION_ORDER)
            for k in range(project_constants.INTERPOLATION_ORDER):
                for l in range(project_constants.INTERPOLATION_ORDER):
                    for x in range(window_size):
                        for y in range(window_size):
                            s[k, l] += grad_I[x, y] * pyramid_I1[level_index][x - k, y - l]
            s_stack.append(s)
        this_b = np.zeros((window_size, window_size), num_corner_pixels)
        for iter_idx in range(max_iter):
            du = np.zeros(I1.shape)
            dv = np.zeros(I1.shape)
            corner_ind = 0
            for corner_pixel in I2_corners:
                i, j = int(corner_pixel[1]), int(corner_pixel[0])
                A = A_stack[corner_ind]
                m_i = find_interpolation_matrix(warped_I2, i, j, order=project_constants.INTERPOLATION_ORDER)
                next_b = r_stack[corner_ind] + np.dot(m_i, s_stack[corner_ind])
                try:
                    out_vector = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, this_b[:, corner_ind]))
                    du[i, j] = out_vector[0]
                    dv[i, j] = out_vector[1]
                    corner_ind += 1
                except np.linalg.LinAlgError:
                    corner_ind += 1
                    continue
                this_b[:, corner_ind] = next_b
            u += du
            v += dv

        if level_index:
            u = 2 * cv2.resize(u, (pyramid_I1[level_index - 1].shape[1], pyramid_I1[level_index - 1].shape[0]))
            v = 2 * cv2.resize(v, (pyramid_I1[level_index - 1].shape[1], pyramid_I1[level_index - 1].shape[0]))

    return u, v


def crop_image(frame: np.ndarray) -> np.ndarray:
    new_frame = frame[:, ~np.isnan(frame).all(axis=0)]  # remove cols
    frame = new_frame[~np.isnan(frame).all(axis=1), :]  # remove rows
    return frame


def stabilize_video(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int) -> None:
    """Calculate LK Optical Flow to stabilize the video and save it to file.

    Args:
        input_video_path: str. path to input video.
        output_video_path: str. path to output stabilized video.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.
        start_rows: int. The number of lines to cut from top.
        end_rows: int. The number of lines to cut from bottom.
        start_cols: int. The number of columns to cut from left.
        end_cols: int. The number of columns to cut from right.

    Returns:
        None.
    """
    cap = cv2.VideoCapture(input_video_path)
    # in order to use tqdm to monitor the progress, we require the number of frames in the video
    num_of_frames = count_frames_manual(cap)
    cv2.destroyAllWindows()
    cap.release()
    cap = cv2.VideoCapture(input_video_path)
    pbar = tqdm(total=num_of_frames)
    vid_params = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = vid_params.get("fps")
    size = vid_params.get("width"), vid_params.get("height")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    isFirstFrame = True
    frame_num = 0
    border_index = int(window_size / 2)
    while cap.isOpened():
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
        else:
            if gray_frame.shape != size:
                gray_frame = cv2.resize(gray_frame, size)
            (new_u, new_v) = faster_lucas_kanade_optical_flow(prev_frame, gray_frame, window_size, max_iter, num_levels)
            h, w = new_u.shape
            u_avg = np.average(new_u[border_index:h - border_index, border_index:w - border_index]) * np.ones(
                new_u.shape)
            v_avg = np.average(new_v[border_index:h - border_index, border_index:w - border_index]) * np.ones(
                new_u.shape)
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
            prev_u = u
            prev_v = v
            prev_frame = gray_frame

        # unlike in HW2, we perform the warp on each RGB channel separately and then merge between them
        if not isFirstFrame:
            (b, g, r) = cv2.split(frame)
            b = warp_image(b, u, v)
            b = crop_image(b)
            if b.shape != size:
                b = cv2.resize(b, size)
            g = warp_image(g, u, v)
            g = crop_image(g)
            if g.shape != size:
                g = cv2.resize(g, size)
            r = warp_image(r, u, v)
            r = crop_image(r)
            if r.shape != size:
                r = cv2.resize(r, size)
            frame = cv2.merge([b, g, r])
        else:
            isFirstFrame = False
        out.write(np.uint8(frame))
        pbar.update(1)
        frame_num = frame_num + 1
        if frame_num == 150:
            break
        if cv2.waitKey(1) == ord('q'):
            break

    pbar.close()
    cv2.destroyAllWindows()
    cap.release()
    out.release()
