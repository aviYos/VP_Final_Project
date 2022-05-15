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


def get_warp(img1, img2, motion=cv2.MOTION_EUCLIDEAN):
    imga = img1.copy().astype(np.float32)
    imgb = img2.copy().astype(np.float32)
    if len(imga.shape) == 3:
        imga = cv2.cvtColor(imga, cv2.COLOR_BGR2GRAY)
    if len(imgb.shape) == 3:
        imgb = cv2.cvtColor(imgb, cv2.COLOR_BGR2GRAY)
    if motion == cv2.MOTION_HOMOGRAPHY:
        warpMatrix=np.eye(3, 3, dtype=np.float32)
    else:
        warpMatrix=np.eye(2, 3, dtype=np.float32)
    warp_matrix = cv2.findTransformECC(templateImage=imga,inputImage=imgb,
                                       warpMatrix=warpMatrix, motionType=motion)[1]
    return warp_matrix


def create_warp_stack(imgs, num_of_frames, motion=cv2.MOTION_EUCLIDEAN):
    warp_stack = []
    print('Creating warp stack...')
    pbar = tqdm(total=num_of_frames-1)
    for i, img in enumerate(imgs[:-1]):
        warp_stack += [get_warp(img, imgs[i+1], motion)]
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
            kernel = signal.gaussian(10000, sigma_mat[i, j])
            kernel = kernel/np.sum(kernel)
            smoothed_trajectory[:, i, j] = ndimage.convolve(original_trajectory[:, i, j], kernel, mode='reflect')
    smoothed_warp = np.apply_along_axis(lambda m:
                                        ndimage.convolve(m, [0, 1, -1], mode='reflect'),
                                        axis=0, arr=smoothed_trajectory)
    return smoothed_warp, smoothed_trajectory, original_trajectory


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
    out.write(np.uint8(images[0]))
    for i in range(num_of_frames-1):
        new_frame = new_images[i]
        if new_frame.size != size:
            new_frame = cv2.resize(new_frame, size)
        out.write(np.uint8(new_frame))

    cv2.destroyAllWindows()
    cap.release()
    out.release()
