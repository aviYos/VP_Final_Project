import cv2
import numpy as np
import project_constants
from tqdm import tqdm
from scipy import signal
from scipy.interpolate import griddata
from harris_corner_detector import our_harris_corner_detector
import os

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
        """  convolve the previous_level_image with PYRAMID_FILTER """
        current_level_image = signal.convolve2d(previous_level_image, project_constants.PYRAMID_FILTER, mode='same', boundary='symm')
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
                out_vector = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))
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

    As for the warping, use `scipy.interpolate`'s `griddata` method. Define the
    grid-points using a flattened version of the `meshgrid` of 0:w-1 and 0:h-1.
    The values here are simply image.flattened().
    The points you wish to interpolate are, again, a flattened version of the
    `meshgrid` matrices - don't forget to add them v and u.
    Use `np.nan` as `griddata`'s fill_value.
    Finally, fill the nan holes with the source image values.
    Hint: For the final step, use np.isnan(image_warp).

    Args:
        image: np.ndarray. Image to warp.
        u: np.ndarray. Optical flow parameters corresponding to the columns.
        v: np.ndarray. Optical flow parameters corresponding to the rows.

    Returns:
        image_warp: np.ndarray. Warped image.
    """

    h, w = image.shape

    """  calculate normalization factor """
    u_normalization_factor = h / u.shape[0]
    v_normalization_factor = w / v.shape[0]

    """ resize u and v to image size """
    resized_u = cv2.resize(u, image.T.shape) * u_normalization_factor
    resized_v = cv2.resize(v, image.T.shape) * v_normalization_factor

    """ create grid """

    x_range = np.linspace(0, w - 1, w)
    y_range = np.linspace(0, h - 1, h)

    x_grid, y_grid = np.meshgrid(x_range, y_range)

    x_grid = x_grid.flatten()
    y_grid = y_grid.flatten()

    """ image values """

    flattened_image = image.flatten()

    """ create grid + u,v """

    grid_x_plus_du = x_grid + resized_u.flatten()
    grid_y_plus_dv = y_grid + resized_v.flatten()

    """ interpolate image """

    interpolated_image = griddata((x_grid, y_grid), flattened_image, (grid_x_plus_du, grid_y_plus_dv), method='cubic',
                                  fill_value=np.nan)

    """ resize interpolated image  """

    interpolated_image_original_size = interpolated_image.reshape(image.shape)

    """ store original values instead of nan values  """

    interpolated_image_original_size[np.isnan(interpolated_image_original_size)] = \
        image[np.isnan(interpolated_image_original_size)]

    image_warp = interpolated_image_original_size

    return image_warp


def lucas_kanade_optical_flow(I1: np.ndarray,
                              I2: np.ndarray,
                              window_size: int,
                              max_iter: int,
                              num_levels: int) -> tuple[np.ndarray, np.ndarray]:
    """Calculate LK Optical Flow for max iterations in num-levels.

    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.
        max_iter: int. Maximal number of LK-steps for each level of the pyramid.
        num_levels: int. Number of pyramid levels.

    Returns:
        (u, v): tuple of np.ndarray-s. Each one of the shape of the
        original image. v encodes the optical flow parameters in rows and u in
        columns.

    Recipe:
        (1) Since the image is going through a series of decimations,
        we would like to resize the image shape to:
        K * (2^(num_levels - 1)) X M * (2^(num_levels - 1)).
        Where: K is the ceil(h / (2^(num_levels - 1)),
        and M is ceil(h / (2^(num_levels - 1)).
        (2) Build pyramids for the two images.
        (3) Initialize u and v as all-zero matrices in the shape of I1.
        (4) For every level in the image pyramid (start from the smallest
        image):
          (4.1) Warp I2 from that level according to the current u and v.
          (4.2) Repeat for num_iterations:
            (4.2.1) Perform a Lucas Kanade Step with the I1 decimated image
            of the current pyramid level and the current I2_warp to get the
            new I2_warp.
          (4.3) For every level which is not the image's level, perform an
          image resize (using cv2.resize) to the next pyramid level resolution
          and scale u and v accordingly.
    """
    h_factor = int(np.ceil(I1.shape[0] / (2 ** (num_levels - 1))))
    w_factor = int(np.ceil(I1.shape[1] / (2 ** (num_levels - 1))))
    IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                  h_factor * (2 ** (num_levels - 1)))
    if I1.shape != IMAGE_SIZE:
        I1 = cv2.resize(I1, IMAGE_SIZE)
    if I2.shape != IMAGE_SIZE:
        I2 = cv2.resize(I2, IMAGE_SIZE)
    # create a pyramid from I1 and I2
    pyramid_I1 = build_pyramid(I1, num_levels)
    pyramid_I2 = build_pyramid(I2, num_levels)

    # start from u and v in the size of smallest image
    u = np.zeros(pyramid_I2[-1].shape)
    v = np.zeros(pyramid_I2[-1].shape)

    for level_index in range(num_levels, -1, -1):

        warped_I2 = warp_image(pyramid_I2[level_index], u, v)

        for iter_idx in range(max_iter):
            du, dv = lucas_kanade_step(pyramid_I1[level_index], warped_I2, window_size)
            u += du
            v += dv
            warped_I2 = warp_image(pyramid_I2[level_index], u, v)

        if level_index:
            u = 2 * cv2.resize(u, (pyramid_I1[level_index - 1].shape[1], pyramid_I1[level_index - 1].shape[0]))
            v = 2 * cv2.resize(v, (pyramid_I1[level_index - 1].shape[1], pyramid_I1[level_index - 1].shape[0]))

    return u, v


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


def our_find_image_corner_pixels(I2: np.ndarray, k: float, threshold:float) -> list[tuple[np.ndarray, np.ndarray]]:
    I2_corners = our_harris_corner_detector(I2, k, threshold)
    corners = np.where(I2_corners == 1)
    corner_pixels = list(zip(corners[1], corners[0]))
    return corner_pixels


def find_image_corner_pixels(I2: np.ndarray, window_size: int, apertureSize: int, k: float) \
        -> list[tuple[np.ndarray, np.ndarray]]:
    I2_corners = cv2.cornerHarris(np.float32(I2), window_size, apertureSize, k)
    # Results are marked through the dilated corners
    dst = cv2.dilate(I2_corners, None)
    corners_logical_mask = dst > 0.001 * dst.max()
    indices = np.where(corners_logical_mask == True)
    corner_pixels = list(zip(indices[0], indices[1]))
    return corner_pixels


def faster_lucas_kanade_step(I1: np.ndarray,
                             I2: np.ndarray,
                             window_size: int, threshold: float, k: float) -> tuple[np.ndarray, np.ndarray]:
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

    I2_corners = our_find_image_corner_pixels(I2, threshold=threshold, k=k)
    # I2_corners = find_image_corner_pixels(I2, window_size, apertureSize=threshold, k=k)

    Ix = signal.convolve2d(I2, project_constants.X_DERIVATIVE_FILTER, mode='same')
    Iy = signal.convolve2d(I2, project_constants.Y_DERIVATIVE_FILTER, mode='same')
    It = I2 - I1

    du = np.zeros(I1.shape)
    dv = np.zeros(I1.shape)

    """ ignore window size / 2 pixels"""
    border_index = int(window_size / 2)

    for corner_pixel in I2_corners:

        i, j = corner_pixel[0], corner_pixel[1]

        Ix_vec = Ix[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()
        Iy_vec = Iy[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()
        It_vec = It[i - border_index:i + border_index + 1, j - border_index:j + border_index + 1].flatten()

        A = np.vstack((Ix_vec, Iy_vec)).T
        b = -1 * It_vec

        """ Compute du and dv by : [du,dv ] = (At * A)^-1 * At * b, and skip if we have A with zero determinant"""
        try:
            out_vector = np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))
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

        for iter_idx in range(max_iter):
            du, dv = faster_lucas_kanade_step(pyramid_I1[level_index], warped_I2, window_size,
                                              project_constants.THRESHOLD, project_constants.K)
            u += du
            v += dv
            warped_I2 = warp_image(pyramid_I2[level_index], u, v)

        if level_index:
            u = 2 * cv2.resize(u, (pyramid_I1[level_index - 1].shape[1], pyramid_I1[level_index - 1].shape[0]))
            v = 2 * cv2.resize(v, (pyramid_I1[level_index - 1].shape[1], pyramid_I1[level_index - 1].shape[0]))

    return u, v


def crop_image(frame: np.ndarray, start_rows: int, end_rows: int, start_cols : int, end_cols: int) -> np.ndarray:
    h, w = frame.shape
    start_rows = min(h, start_rows)
    start_cols = min(w, start_cols)
    end_rows = min(h, end_rows)
    end_cols = min(w, end_cols)
    return frame[start_rows:h-end_rows, start_cols:w-end_cols]

def stabilize_video(
        input_video_path: str, output_video_path: str, window_size: int,
        max_iter: int, num_levels: int, start_rows: int = 10,
        start_cols: int = 2, end_rows: int = 30, end_cols: int = 30) -> None:
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
            isFirstFrame = False
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
        (b, g, r) = cv2.split(frame)
        b = warp_image(b, u, v)
        b = crop_image(b, start_rows, end_rows, start_cols, end_cols)
        if b.shape != size:
            b = cv2.resize(b, size)
        g = warp_image(g, u, v)
        g = crop_image(g, start_rows, end_rows, start_cols, end_cols)
        if g.shape != size:
            g = cv2.resize(g, size)
        r = warp_image(r, u, v)
        r = crop_image(r, start_rows, end_rows, start_cols, end_cols)
        if r.shape != size:
            r = cv2.resize(r, size)
        frame = cv2.merge([b, g, r])
        out.write(np.uint8(frame))
        pbar.update(1)
        frame_num = frame_num + 1
        if frame_num == 2:
            break
        if cv2.waitKey(1) == ord('q'):
            break

    pbar.close()
    cv2.destroyAllWindows()
    cap.release()
    out.release()
