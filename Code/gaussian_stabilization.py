import numpy as np
import cv2
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


def build_pyramid(image: np.ndarray, num_levels: int) -> list[np.ndarray]:
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


def warp_image(image: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
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
    """
    Args:
        I1: np.ndarray. Image at time t.
        I2: np.ndarray. Image at time t+1.
        window_size: int. The window is of shape window_size X window_size.

    Returns:
        (du, dv): tuple of np.ndarray-s. Each one of the shape of the
        original image. dv encodes the shift in rows and du in columns.
    """

    h, w = I1.shape

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


def lucas_kanade(imgs, num_of_frames, num_levels):
    print('Performing Lucas Kanade...')
    pbar = tqdm(total=num_of_frames)
    border_index = int(project_constants.WINDOW_SIZE_TAU / 2)
    isFirstFrame = True
    u_v_list = []
    for i, img in enumerate(imgs):
        if len(img.shape) == 3:
            gray_frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if isFirstFrame:
            h_factor = int(np.ceil(gray_frame.shape[0] / (2 ** (num_levels - 1))))
            w_factor = int(np.ceil(gray_frame.shape[1] / (2 ** (num_levels - 1))))
            IMAGE_SIZE = (w_factor * (2 ** (num_levels - 1)),
                         h_factor * (2 ** (num_levels - 1)))

            u = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
            v = np.zeros((IMAGE_SIZE[1], IMAGE_SIZE[0]))
            prev_u = u
            prev_v = v
            isFirstFrame = False
            prev_frame = gray_frame
        else:
            new_u, new_v = faster_lucas_kanade_optical_flow(prev_frame, gray_frame, project_constants.WINDOW_SIZE_TAU,
                                                             project_constants.MAX_ITER_TAU, project_constants.NUM_LEVELS_TAU)
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
            prev_u = u
            prev_v = v
            prev_frame = gray_frame
        pbar.update(1)
        h, w, _ = imgs[0].shape
        u_avg = u[int(h / 2), int(w / 2)]
        v_avg = v[int(h / 2), int(w / 2)]
        u_v_list.append((u_avg, v_avg))
    pbar.close()
    return u_v_list


def get_warp(img1, img2, motion=cv2.MOTION_EUCLIDEAN, u_v_list=None, index=0):
    imga = img1.copy().astype(np.float32)
    imgb = img2.copy().astype(np.float32)
    if len(imga.shape) == 3:
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
    if len(imgb.shape) == 3:
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warpMatrix = np.eye(3, 3, dtype=np.float32)
    else:
        warpMatrix = np.eye(2, 3, dtype=np.float32)
    if project_constants.USE_LK:
        warpMatrix[0, 2] = u_v_list[index][0]
        warpMatrix[1, 2] = u_v_list[index][1]
    warp_matrix = cv2.findTransformECC(templateImage=imga, inputImage=imgb,
                                       warpMatrix=warpMatrix, motionType=motion, criteria=project_constants.criteria)[1]
    return warp_matrix


def create_warp_stack(imgs, num_of_frames, motion=cv2.MOTION_EUCLIDEAN):
    warp_stack = []
    if project_constants.USE_LK:
        u_v_list = lucas_kanade(imgs, num_of_frames, project_constants.NUM_LEVELS_TAU)
    print('Creating warp stack...')
    pbar = tqdm(total=num_of_frames - 1)
    for i, img in enumerate(imgs[:-1]):
        warp_stack += [get_warp(img, imgs[i+1], motion, u_v_list, i+1)]
        pbar.update(1)
    pbar.close()
    return np.array(warp_stack)


def homography_gen(warp_stack):
    H_tot = np.eye(3)
    wsp = np.dstack([warp_stack[:, 0, :], warp_stack[:, 1, :], np.array([[0, 0, 1]]*warp_stack.shape[0])])
    for i in range(len(warp_stack)):
        H_tot = np.matmul(wsp[i].T, H_tot)
        yield np.linalg.inv(H_tot)


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


def gauss_convolve(trajectory, window, sigma):
    kernel = signal.gaussian(window, std=sigma)
    kernel = kernel/np.sum(kernel)
    return ndimage.convolve(trajectory, kernel, mode='reflect')


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


def gaussian_stabilization(
        input_video_path: str, output_video_path: str, motion=cv2.MOTION_EUCLIDEAN) -> None:
    cap = cv2.VideoCapture(input_video_path)
    # in order to use tqdm to monitor the progress, we require the number of frames in the video
    num_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_params = get_video_parameters(cap)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = vid_params.get("fps")
    size = vid_params.get("width"), vid_params.get("height")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    images = []
    if motion == cv2.MOTION_HOMOGRAPHY:
        sigma_mat = project_constants.sigma_mat_3D
    else:
        sigma_mat = project_constants.sigma_mat_2D
    for i in range(num_of_frames):
        ret, frame = cap.read()
        if not ret:
            break
        images += [frame]
    warp_stack = create_warp_stack(images, num_of_frames, motion=motion)
    smoothed_warp, smoothed_trajectory, original_trajectory = moving_average(warp_stack,
                                                                             sigma_mat=sigma_mat)
    print('Applying warp...')
    new_images = apply_warping_fullview(images=images, warp_stack=warp_stack - smoothed_warp,
                                        num_of_frames=num_of_frames)
    first_frame = images[0]
    first_frame = add_borders(first_frame, project_constants.START_ROWS, project_constants.END_ROWS,
                              project_constants.START_COLS, project_constants.END_COLS)
    out.write(np.uint8(first_frame))
    for i in range(num_of_frames-1):
        new_frame = new_images[i]
        if new_frame.size != size:
            new_frame = cv2.resize(new_frame, size)
        new_frame = add_borders(new_frame, project_constants.START_ROWS, project_constants.END_ROWS,
                                project_constants.START_COLS, project_constants.END_COLS)
        out.write(np.uint8(new_frame))
    cv2.destroyAllWindows()
    cap.release()
    out.release()
