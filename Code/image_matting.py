import cv2
import GeodisTK
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from tqdm import tqdm
import project_constants
import project_utils


class image_matting:

    def __init__(self):
        self.binary_video_cap = cv2.VideoCapture(project_constants.BINARY_PATH)
        self.extracted_video_cap = cv2.VideoCapture(project_constants.EXTRACTED_PATH)
        self.logger = project_utils.create_general_logger()
        self.number_of_frames = int(self.extracted_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(self.extracted_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) \
                                / project_constants.resize_factor)
        self.frame_width = int(self.extracted_video_cap.get(cv2.CAP_PROP_FRAME_WIDTH) / project_constants.resize_factor)
        self.fps = project_utils.get_video_fps(self.extracted_video_cap)
        self.output_frame_width = int(self.frame_width * project_constants.resize_factor)
        self.output_frame_height = int(self.frame_height * project_constants.resize_factor)
        self.alpha_video_writer = None
        self.matted_video_writer = None
        self.progress_bar = tqdm(total=self.number_of_frames)
        self.distance_map_radius = project_constants.distance_map_radius
        self.background_image = self.load_background_image()

    def create_video_writers(self):

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.alpha_video_writer = cv2.VideoWriter(project_constants.ALPHA_PATH, fourcc, self.fps, (
            self.output_frame_width,
            self.output_frame_height))

        self.matted_video_writer = cv2.VideoWriter(project_constants.MATTED_PATH, fourcc, self.fps, (
            self.output_frame_width,
            self.output_frame_height))

    @staticmethod
    def normalize_frame(frame):
        try:
            normalized_frame = (1. * frame - np.amin(frame)) / (np.amax(frame) - np.amin(frame)) * 255
            return normalized_frame
        except Exception as e:
            print(e)

    @staticmethod
    def create_distance_map(Normalized_FG_Pr_map, Normalized_BG_Pr_map, foreground_logical_matrix,
                            background_logical_matrix):
        foreground_distance_map = GeodisTK.geodesic2d_fast_marching(Normalized_FG_Pr_map.astype('float32'),
                                                                    foreground_logical_matrix)
        background_distance_map = GeodisTK.geodesic2d_fast_marching(Normalized_BG_Pr_map.astype('float32'),
                                                                    background_logical_matrix)
        return foreground_distance_map, background_distance_map

    @staticmethod
    def create_foreground_background_pixels_map(binary_frame):
        foreground_logical_matrix = (binary_frame > 150).astype(np.uint8)
        background_logical_matrix = (binary_frame <= 150).astype(np.uint8)
        return foreground_logical_matrix, background_logical_matrix

    def load_background_image(self):
        background_image = plt.imread(project_constants.BACKGROUND_IMAGE_PATH)
        background_image = cv2.resize(background_image, (self.frame_width, self.frame_height))
        background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)
        return background_image

    def create_alpha_frame_from_trimap(self, trimap, trimap_mask, Wf, Wb, foreground_distance_map,
                                       background_distance_map, bound_rect=None, is_first_frame_flag=False):

        if is_first_frame_flag:
            alpha = trimap.copy()
            alpha[trimap_mask] = Wf[trimap_mask] / (Wf[trimap_mask] + Wb[trimap_mask])
            alpha[foreground_distance_map == 0] = 1
            alpha[background_distance_map == 0] = 0
        else:
            alpha_bounding_rect = trimap.copy()
            alpha_bounding_rect[trimap_mask] = Wf[trimap_mask] / (Wf[trimap_mask] + Wb[trimap_mask])
            alpha_bounding_rect[foreground_distance_map == 0] = 1
            alpha_bounding_rect[background_distance_map == 0] = 0
            alpha = np.zeros((self.frame_height, self.frame_width))
            alpha[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]] = \
                alpha_bounding_rect

        alpha = np.atleast_3d(alpha)
        alpha = cv2.merge([alpha, alpha, alpha])
        return alpha

    def create_probability_map(self, foreground_logical_matrix, background_logical_matrix, value_channel,
                               P_F_given_c=None, P_B_given_c=None, is_first_frame=False):
        x_grid = np.linspace(0, 255, 256)

        if is_first_frame:
            kde_foreground = gaussian_kde(value_channel[np.where(foreground_logical_matrix == 1)],
                                          bw_method='silverman')
            kde_foreground_pdf = kde_foreground.evaluate(x_grid)

            kde_bg = gaussian_kde(value_channel[np.where(background_logical_matrix == 1)], bw_method='silverman')
            kde_bg_pdf = kde_bg.evaluate(x_grid)

            # probabilties of background and foreground
            P_F_given_c = kde_foreground_pdf / (kde_foreground_pdf + kde_bg_pdf)
            P_B_given_c = kde_bg_pdf / (kde_foreground_pdf + kde_bg_pdf)

        # probabilities map of background and foreground
        foreground_probability_map = P_F_given_c[value_channel]
        background_probability_map = P_B_given_c[value_channel]

        normalized_foreground_probability_map = self.normalize_frame(foreground_probability_map)
        normalized_background_probability_map = self.normalize_frame(background_probability_map)

        return foreground_probability_map, background_probability_map, normalized_foreground_probability_map, normalized_background_probability_map, P_F_given_c, P_B_given_c

    @staticmethod
    def create_delta(Vf):
        se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        BWerode = cv2.morphologyEx(Vf, cv2.MORPH_ERODE, se1)
        delta = (255 * (np.abs(BWerode - Vf) > 0)).astype('uint8')
        return delta

    @staticmethod
    def create_Vf_and_Vb(foreground_distance_map, background_distance_map, current_shape):
        Vf = np.zeros(current_shape)
        Vb = np.zeros(current_shape)
        Vf[(foreground_distance_map.astype('int') - background_distance_map.astype('int')) <= 0] = 255
        Vb[(background_distance_map.astype('int') - foreground_distance_map.astype('int')) <= 0] = 255
        return Vf, Vb

    def create_trimap_first_frame(self, foreground_distance_map, background_distance_map):

        current_shape = (self.frame_height, self.frame_width)
        Vf, Vb = self.create_Vf_and_Vb(foreground_distance_map, background_distance_map, current_shape)

        delta = self.create_delta(Vf)

        # trimap
        narrow_band = cv2.morphologyEx(delta, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        trimap = np.zeros((self.frame_height, self.frame_width))
        trimap[(Vf == 255) & (narrow_band == 0)] = 1
        trimap[(Vb == 255) & (narrow_band == 0)] = 0
        trimap[narrow_band == 255] = 0.5  # undecided region
        return trimap

    def create_trimap(self, foreground_distance_map, background_distance_map):

        current_shape = foreground_distance_map.shape
        Vf, Vb = self.create_Vf_and_Vb(foreground_distance_map, background_distance_map, current_shape)

        delta = self.create_delta(Vf)
        # union of euclidean balls with radius rho
        narrow_band = cv2.morphologyEx(delta, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        trimap = np.zeros(foreground_distance_map.shape)
        trimap[(Vf == 255) & (narrow_band == 0)] = 1
        trimap[(Vb == 255) & (narrow_band == 0)] = 0
        trimap[narrow_band == 255] = 0.5  # undecided region
        return trimap

    def create_Wf_Wb(self, trimap_mask, foreground_distance_map, background_distance_map, foreground_probability_map,
                     background_probability_map, current_size):

        Wf = np.zeros(current_size)
        Wb = np.zeros(current_size)

        Wf[trimap_mask] = ((foreground_distance_map[trimap_mask] / 1.0) **
                           (-1 * self.distance_map_radius)) * foreground_probability_map[
                              trimap_mask]
        Wb[trimap_mask] = ((background_distance_map[trimap_mask] / 1.0) **
                           (-1 * self.distance_map_radius)) * background_probability_map[
                              trimap_mask]

        return Wf, Wb

    def handle_first_frame(self):

        _, extracted_Frame = self.extracted_video_cap.read()
        extracted_Frame = cv2.resize(extracted_Frame, (self.frame_width, self.frame_height))

        _, binary_frame = self.binary_video_cap.read()
        binary_frame = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
        binary_frame = cv2.resize(binary_frame, (self.frame_width, self.frame_height))

        foreground_logical_matrix, background_logical_matrix = self.create_foreground_background_pixels_map(
            binary_frame)

        _, _, value_channel = cv2.split(cv2.cvtColor(extracted_Frame, cv2.COLOR_BGR2HSV))

        foreground_probability_map, background_probability_map, normalized_foreground_probability_map, normalized_background_probability_map, P_F_given_c, P_B_given_c \
            = self.create_probability_map(foreground_logical_matrix, background_logical_matrix, value_channel,
                                          is_first_frame=True)

        foreground_distance_map, background_distance_map = self.create_distance_map(
            normalized_foreground_probability_map,
            normalized_background_probability_map,
            foreground_logical_matrix,
            background_logical_matrix)

        trimap = self.create_trimap_first_frame(foreground_distance_map, background_distance_map)

        # alpha + matting
        trimap_mask = ((trimap == 0.5) & (foreground_distance_map != 0) & (background_distance_map != 0))

        current_size = (self.frame_height, self.frame_width)
        Wf, Wb = self.create_Wf_Wb(trimap_mask, foreground_distance_map, background_distance_map,
                                   foreground_probability_map,
                                   background_probability_map, current_size)

        alpha = self.create_alpha_frame_from_trimap(trimap, trimap_mask, Wf, Wb, foreground_distance_map,
                                                    background_distance_map, is_first_frame_flag=True)

        matted_frame = alpha * extracted_Frame.astype('float') + (1 - alpha) * self.background_image.astype('float')
        matted_frame = matted_frame.astype('uint8')

        matted_frame_1080p = cv2.resize(matted_frame, (int(self.output_frame_width), int(self.output_frame_height)))

        alpha_1080p = cv2.resize((self.normalize_frame(alpha)).astype('uint8'),
                                 (int(self.output_frame_width), int(self.output_frame_height)))

        self.matted_video_writer.write(matted_frame_1080p)
        self.alpha_video_writer.write(alpha_1080p)
        self.progress_bar.update(1)

        return P_F_given_c, P_B_given_c

    def create_matted_and_alpha_video(self, P_F_given_c, P_B_given_c):

        # loop for creating following frames for matted vid #
        for i in range(self.number_of_frames - 1):
            _, extracted_frame = self.extracted_video_cap.read()
            if extracted_frame is None:
                break
            else:
                extracted_frame = cv2.resize(extracted_frame, (self.frame_width, self.frame_height))

            _, binary_frame = self.binary_video_cap.read()
            binary_frame = cv2.resize(binary_frame, (self.frame_width, self.frame_height))
            _, binary_frame = cv2.threshold(binary_frame[:, :, 0], 0, 255, cv2.THRESH_OTSU)  # avoid noise

            bound_rect = cv2.boundingRect(binary_frame)

            binary_frame = project_utils.slice_frame_from_bounding_rect(binary_frame, bound_rect)
            if not binary_frame.shape[0]:
                continue

            value_channel = cv2.split(cv2.cvtColor(extracted_frame, cv2.COLOR_BGR2HSV))[2]
            value_channel = project_utils.slice_frame_from_bounding_rect(value_channel, bound_rect)

            foreground_logical_matrix, background_logical_matrix = self.create_foreground_background_pixels_map(
                binary_frame)

            foreground_probability_map, background_probability_map, normalized_foreground_probability_map, \
            normalized_background_probability_map, _, _ = self.create_probability_map(
                foreground_logical_matrix,
                background_logical_matrix, value_channel, P_F_given_c, P_B_given_c, is_first_frame=False)

            fg_prob_grad = np.sqrt(
                (cv2.Sobel(normalized_foreground_probability_map, cv2.CV_64F, 1, 0, ksize=5)) ** 2 + (
                    cv2.Sobel(normalized_foreground_probability_map, cv2.CV_64F, 0, 1, ksize=5)) ** 2)
            fg_prob_grad = self.normalize_frame(fg_prob_grad)

            # binary frame to seeds conversion #
            binary_frame_fg = cv2.morphologyEx(binary_frame, cv2.MORPH_ERODE,
                                               cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5)).T, iterations=2)
            binary_frame_bg = cv2.morphologyEx(binary_frame, cv2.MORPH_DILATE,
                                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=6)
            binary_frame_bg = np.bitwise_not(binary_frame_bg)

            foreground_distance_map = GeodisTK.geodesic2d_fast_marching(fg_prob_grad.astype('float32'), binary_frame_fg)
            background_distance_map = GeodisTK.geodesic2d_fast_marching(fg_prob_grad.astype('float32'), binary_frame_bg)

            trimap = self.create_trimap(foreground_distance_map, background_distance_map)

            trimap_mask = ((trimap == 0.5) & (foreground_distance_map != 0) & (background_distance_map != 0))

            current_size = foreground_distance_map.shape
            Wf, Wb = self.create_Wf_Wb(trimap_mask, foreground_distance_map, background_distance_map,
                                       foreground_probability_map,
                                       background_probability_map, current_size)

            alpha = self.create_alpha_frame_from_trimap(trimap, trimap_mask, Wf, Wb, foreground_distance_map,
                                                        background_distance_map, bound_rect)

            alpha_original_size = cv2.resize((self.normalize_frame(alpha)).astype('uint8'),
                                             (self.output_frame_width, self.output_frame_height))

            matted_frame = alpha * extracted_frame.astype('float') + (1 - alpha) * self.background_image.astype('float')

            matted_frame_original_size = cv2.resize(matted_frame, (self.output_frame_width, self.output_frame_height))

            self.matted_video_writer.write(matted_frame_original_size.astype('uint8'))

            self.alpha_video_writer.write(alpha_original_size)

            self.progress_bar.update(1)

    def close_all_videos(self):

        self.extracted_video_cap.release()
        self.matted_video_writer.release()
        self.alpha_video_writer.release()
        self.binary_video_cap.release()
        cv2.destroyAllWindows()
        self.progress_bar.close()

    def main_image_matting_module(self):

        self.create_video_writers()

        P_F_given_c, P_B_given_c = self.handle_first_frame()

        self.create_matted_and_alpha_video(P_F_given_c, P_B_given_c)

        self.close_all_videos()


if __name__ == '__main__':
    # matting_module()
    mat = image_matting()
    mat.main_image_matting_module()
