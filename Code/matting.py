import cv2
import GeodisTK
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import config as cfg
import time
from tqdm import tqdm

def normalize_im(I):
    normed_im = (1. * I - np.amin(I)) / (np.amax(I) - np.amin(I))
    return normed_im

def define_bounding_rect(I, bound_rect):
    return I[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]]

def inverse_fix_border(frame_s):
    frame = frame_s.copy()
    s = frame.shape
    T = cv2.getRotationMatrix2D((s[1] / 2, s[0] / 2), 0, 1.04)
    T_inv = cv2.invertAffineTransform(T)
    frame = cv2.warpAffine(frame, T_inv, (s[1], s[0]))
    return frame

def inverse_smooth_transform(frame_s, transforms_smooth, i):
    frame = frame_s.copy()
    dx = transforms_smooth[i, 0]
    dy = transforms_smooth[i, 1]
    da = transforms_smooth[i, 2]

    m = np.zeros((2, 3), np.float32)
    m[0, 0] = np.cos(da)
    m[0, 1] = -np.sin(da)
    m[1, 0] = np.sin(da)
    m[1, 1] = np.cos(da)
    m[0, 2] = dx
    m[1, 2] = dy

    inverse_affine = cv2.invertAffineTransform(m)
    frame = cv2.warpAffine(frame, inverse_affine, (frame.shape[1], frame.shape[0]))

    return frame

def matting_module():

    start_time = time.time()

    transforms_smooth = np.load(cfg.path_to_transformations_list) # tranforms file from stabilization stage

    binary_vid = cv2.VideoCapture(cfg.path_to_binary)
    _, binary_frame = binary_vid.read()#first frame is not used

    stabilized_vid = cv2.VideoCapture(cfg.path_to_stabilized)
    width = int(cv2.VideoCapture.get(stabilized_vid, cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cv2.VideoCapture.get(stabilized_vid, cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cv2.VideoCapture.get(stabilized_vid, cv2.CAP_PROP_FRAME_COUNT))
    fps_input = int(cv2.VideoCapture.get(stabilized_vid, cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    matted = cv2.VideoWriter(cfg.path_to_matted, fourcc, fps_input,(int(width * cfg.resize_factor), int(height * cfg.resize_factor)))
    alpha_vid = cv2.VideoWriter(cfg.path_to_alpha, fourcc, fps_input,(int(width * cfg.resize_factor), int(height * cfg.resize_factor)))
    unstabilized_alpha_vid = cv2.VideoWriter(cfg.path_to_unstabilized_alpha, fourcc, fps_input,(int(width * cfg.resize_factor), int(height * cfg.resize_factor)))

    output_width = int(width * cfg.resize_factor)
    output_height = int(height * cfg.resize_factor)

    # acquire seeds for first frame

    FG_scrib = plt.imread(cfg.path_to_FG_scribbles)
    BG_scrib = plt.imread(cfg.path_to_BG_scribbles)
    pbar = tqdm(total=frame_count)

    # In order to get the scribbles easily for the first frame we use the fact that R_value (Red) = 249 is not present
    # in the histogram of the first frame. Also, this is the only time when we build KDE, as it serves us for the
    # rest of the frames under the assumption that p(F|c) and p(B|c) are constant for all frames

    RED_val = cfg.RED_value  # this color is not present in the first frame's histogram
    R_FG, G_FG, B_FG = FG_scrib[:, :, 0], FG_scrib[:, :, 1], FG_scrib[:, :, 2]
    R_BG, G_BG, B_BG = BG_scrib[:, :, 0], BG_scrib[:, :, 1], BG_scrib[:, :, 2]

    row_index = np.transpose(np.atleast_2d(np.arange(height))) @ np.ones((1, width))  # coordinate matrix containing row number in index (i,j)
    column_index = np.ones((height, 1)) @ np.atleast_2d(np.arange(width))  # coordinate matrix containing column number in index (i,j)

    row_scrib_FG = np.atleast_2d(row_index[(R_FG == RED_val) & (B_FG == 0) & (G_FG == 0)])
    col_scrib_FG = np.atleast_2d(column_index[(R_FG == RED_val) & (B_FG == 0) & (G_FG == 0)])
    fg_seeds_binary_matrix = np.zeros((height, width), np.uint8)
    fg_seeds_binary_matrix[row_scrib_FG[:].astype('uint16'), col_scrib_FG[:].astype('uint16')] = 1
    row_scrib_FG = row_scrib_FG.T
    col_scrib_FG = col_scrib_FG.T
    fg_seeds = (np.concatenate((row_scrib_FG, col_scrib_FG), axis=1)).astype('int')

    row_scrib_BG = np.atleast_2d(row_index[(R_BG == RED_val) & (B_BG == 0) & (G_BG == 0)])
    col_scrib_BG = np.atleast_2d(column_index[(R_BG == RED_val) & (B_BG == 0) & (G_BG == 0)])
    bg_seeds_binary_matrix = np.zeros((height, width), np.uint8)
    bg_seeds_binary_matrix[row_scrib_BG[:].astype('uint16'), col_scrib_BG[:].astype('uint16')] = 1
    row_scrib_BG = row_scrib_BG.T
    col_scrib_BG = col_scrib_BG.T
    bg_seeds = (np.concatenate((row_scrib_BG, col_scrib_BG), axis=1)).astype('int')

    #####FIRST FRAME - READ->CONVERT_BGR2HSV->GET_SCRIBBLES->KDE->DISTANCE_MAP->MATTING->WRITE2MATTED.AVI#####
    success, im = stabilized_vid.read()
    if not success:
        print("Error while trying to read first frame.\n")

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # frame read is BGR format because we use cv2
    im_V = im_hsv[:, :, 2]

    # calculate KDE for background and foreground from scribbles
    x_grid = np.linspace(0, 255, 256)

    kde_fg = gaussian_kde(im_V[fg_seeds[:, 0], fg_seeds[:, 1]], bw_method='silverman')
    kde_fg_pdf = kde_fg.evaluate(x_grid)

    kde_bg = gaussian_kde(im_V[bg_seeds[:, 0], bg_seeds[:, 1]], bw_method='silverman')
    kde_bg_pdf = kde_bg.evaluate(x_grid)

    # probabilties of background and foreground
    P_F_given_c = kde_fg_pdf / (kde_fg_pdf + kde_bg_pdf)
    P_B_given_c = kde_bg_pdf / (kde_fg_pdf + kde_bg_pdf)

    Vf = np.zeros((height, width))
    Vb = np.zeros((height, width))

    trimap = np.zeros((height, width))

    r = cfg.r
    Wf = np.zeros((height, width))
    Wb = np.zeros((height, width))

    # change new background dimensions #
    new_BG = plt.imread(r"D:\VideoProcessing\Final_Project\FinalProject\Inputs\background.jpg")
    new_BG = cv2.resize(new_BG, (width, height))
    new_BG = cv2.cvtColor(new_BG, cv2.COLOR_RGB2BGR)
    # change new background dimensions #

    # probabilities map of background and foreground
    FG_Pr_map = P_F_given_c[im_V]
    BG_Pr_map = P_B_given_c[im_V]

    Normalized_FG_Pr_map = normalize_im(FG_Pr_map) * 255.
    Normalized_BG_Pr_map = normalize_im(BG_Pr_map) * 255.

    # compute distance map
    fg_dmap = GeodisTK.geodesic2d_fast_marching(Normalized_FG_Pr_map.astype('float32'), fg_seeds_binary_matrix)
    bg_dmap = GeodisTK.geodesic2d_fast_marching(Normalized_BG_Pr_map.astype('float32'), bg_seeds_binary_matrix)

    Vf[(fg_dmap.astype('int') - bg_dmap.astype('int')) <= 0] = 255
    Vb[(bg_dmap.astype('int') - fg_dmap.astype('int')) <= 0] = 255

    # delta
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    BWerode = cv2.morphologyEx(Vf, cv2.MORPH_ERODE, se1)
    delta = (255 * (np.abs(BWerode - Vf) > 0)).astype('uint8')

    # union of euclidean balls with radius rho
    rho = cfg.rho1  # expresses the amount of uncertainty in the narrow band refinement process
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rho, rho))
    narrow_band = cv2.morphologyEx(delta, cv2.MORPH_DILATE, se2)

    # trimap
    trimap[(Vf == 255) & (narrow_band == 0)] = 1
    trimap[(Vb == 255) & (narrow_band == 0)] = 0
    trimap[narrow_band == 255] = 0.5  # undecided region

    # alpha + matting
    trimap_mask = ((trimap == 0.5) & (fg_dmap != 0) & (bg_dmap != 0))

    Wf[trimap_mask] = ((fg_dmap[trimap_mask] / 1.0) ** (-r)) * FG_Pr_map[trimap_mask]
    Wb[trimap_mask] = ((bg_dmap[trimap_mask] / 1.0) ** (-r)) * BG_Pr_map[trimap_mask]
    alpha = trimap.copy()
    alpha[trimap_mask] = Wf[trimap_mask] / (Wf[trimap_mask] + Wb[trimap_mask])
    alpha[fg_dmap == 0] = 1
    alpha[bg_dmap == 0] = 0
    alpha = np.atleast_3d(alpha)
    alpha = cv2.merge([alpha, alpha, alpha])

    matted_frame = alpha * im.astype('float') + (1 - alpha) * new_BG.astype('float')
    matted_frame = matted_frame.astype('uint8')

    matted_frame_1080p = cv2.resize(matted_frame, (int(output_width), int(output_height)))

    # build unstabilized_alpha #
    unstabilized_alpha = inverse_fix_border(alpha)
    unstabilized_alpha = inverse_smooth_transform(unstabilized_alpha, transforms_smooth, i=0)
    unstabilized_alpha_1080p = cv2.resize((normalize_im(unstabilized_alpha)*255.).astype('uint8'), (int(output_width), int(output_height)))

    alpha_1080p = cv2.resize((normalize_im(alpha)*255.).astype('uint8'), (int(output_width), int(output_height)))

    matted.write(matted_frame_1080p)
    alpha_vid.write(alpha_1080p)
    unstabilized_alpha_vid.write(unstabilized_alpha_1080p)
    pbar.update(1)
    #####FIRST FRAME - READ->CONVERT_BGR2HSV->GET_SCRIBBLES->KDE->DISTANCE_MAP->MATTING->WRITE2MATTED.AVI#####

    se_binary_fg = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5)).T
    se_binary_bg = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    rho = cfg.rho2
    se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    se2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (rho, rho))

    # loop for creating following frames for matted vid #
    for i in range(frame_count - 1):
        ret, im = stabilized_vid.read()
        if ret == False:
            break

        _, binary_frame = binary_vid.read()
        otsu_thres, binary_frame = cv2.threshold(binary_frame[:, :, 0], 0, 255, cv2.THRESH_OTSU)  # avoid noise

        bound_rect = cv2.boundingRect(binary_frame)

        binary_frame = define_bounding_rect(binary_frame, bound_rect)

        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # frame read is BGR format because we use cv2
        im_V = im_hsv[:, :, 2]
        im_V = define_bounding_rect(im_V, bound_rect)

        FG_Pr_map = P_F_given_c[im_V]
        BG_Pr_map = P_B_given_c[im_V]

        FG_Pr_map = normalize_im(FG_Pr_map) * 255.
        BG_Pr_map = normalize_im(BG_Pr_map) * 255.

        fg_prob_grad = np.sqrt((cv2.Sobel(FG_Pr_map, cv2.CV_64F, 1, 0, ksize=5)) ** 2 + (cv2.Sobel(FG_Pr_map, cv2.CV_64F, 0, 1, ksize=5)) ** 2)
        fg_prob_grad = normalize_im(fg_prob_grad) * 255.

        # binary frame to seeds conversion #
        binary_frame_fg = cv2.morphologyEx(binary_frame, cv2.MORPH_ERODE, se_binary_fg, iterations=2)
        binary_frame_bg = cv2.morphologyEx(binary_frame, cv2.MORPH_DILATE, se_binary_bg, iterations=6)
        binary_frame_bg = np.bitwise_not(binary_frame_bg)
        # binary frame to seeds conversion #

        fg_dmap = GeodisTK.geodesic2d_fast_marching(fg_prob_grad.astype('float32'), binary_frame_fg)
        bg_dmap = GeodisTK.geodesic2d_fast_marching(fg_prob_grad.astype('float32'), binary_frame_bg)

        Vf = np.zeros(fg_dmap.shape)
        Vb = np.zeros(fg_dmap.shape)
        Wf = np.zeros(fg_dmap.shape)
        Wb = np.zeros(fg_dmap.shape)
        trimap = np.zeros(fg_dmap.shape)

        Vf[(fg_dmap.astype('int') - bg_dmap.astype('int')) <= 0] = 255
        Vb[(bg_dmap.astype('int') - fg_dmap.astype('int')) <= 0] = 255

        # delta
        BWerode = cv2.morphologyEx(Vf, cv2.MORPH_ERODE, se1)
        delta = (255 * (np.abs(BWerode - Vf) > 0)).astype('uint8')

        # union of euclidean balls with radius rho
        narrow_band = cv2.morphologyEx(delta, cv2.MORPH_DILATE, se2)

        trimap[(Vf == 255) & (narrow_band == 0)] = 1
        trimap[(Vb == 255) & (narrow_band == 0)] = 0
        trimap[narrow_band == 255] = 0.5  # undecided region

        trimap_mask = ((trimap == 0.5) & (fg_dmap != 0) & (bg_dmap != 0))

        Wf[trimap_mask] = ((fg_dmap[trimap_mask] / 1.0) ** (-r)) * FG_Pr_map[trimap_mask]
        Wb[trimap_mask] = ((bg_dmap[trimap_mask] / 1.0) ** (-r)) * BG_Pr_map[trimap_mask]
        alpha_bounding_rect = trimap.copy()
        alpha_bounding_rect[trimap_mask] = Wf[trimap_mask] / (Wf[trimap_mask] + Wb[trimap_mask])
        alpha_bounding_rect[fg_dmap == 0] = 1
        alpha_bounding_rect[bg_dmap == 0] = 0
        alpha = np.zeros((np.shape(im)[0], np.shape(im)[1]))
        alpha[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]] = alpha_bounding_rect
        alpha = np.atleast_3d(alpha)
        alpha = cv2.merge([alpha, alpha, alpha])

        alpha_1080p = cv2.resize((normalize_im(alpha) * 255.).astype('uint8'), (output_width, output_height))

        unstabilized_alpha = inverse_fix_border(alpha)
        unstabilized_alpha = inverse_smooth_transform(unstabilized_alpha, transforms_smooth, i=i+1)
        unstabilized_alpha_1080p = cv2.resize((normalize_im(unstabilized_alpha) * 255.).astype('uint8'), (output_width, output_height))
        unstabilized_alpha_vid.write(unstabilized_alpha_1080p)

        matted_frame = alpha * im.astype('float') + (1 - alpha) * new_BG.astype('float')

        matted_frame_1080p = cv2.resize(matted_frame, (output_width, output_height))

        matted.write((matted_frame_1080p).astype('uint8'))

        alpha_vid.write(alpha_1080p)

        pbar.update(1)


    stabilized_vid.release()
    matted.release()
    unstabilized_alpha_vid.release()
    alpha_vid.release()
    cv2.destroyAllWindows()
    pbar.close()

    cfg.write2log('matting', time.time() - start_time)

if __name__ == "__main__":
    matting_module()