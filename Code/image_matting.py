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
        self.stabilized_video_cap = cv2.VideoCapture(project_constants.STABILIZE_PATH)
        self.binary_video_cap = cv2.VideoCapture(project_constants.BINARY_PATH)
        self.extracted_video_cap = cv2.VideoCapture(project_constants.EXTRACTED_PATH)
        self.logger = project_utils.create_general_logger()
        self.number_of_frames = int(self.extracted_video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(
            self.extracted_video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / project_constants.resize_factor)
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
    def create_foreground_background_pixels_map(binary_frame):
        foreground_logical_matrix = (binary_frame > 200).astype(np.uint8)
        background_logical_matrix = (binary_frame <= 200).astype(np.uint8)
        eroded_foreground = cv2.morphologyEx(foreground_logical_matrix, cv2.MORPH_ERODE,
                                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)),
                                             iterations=3)
        dilated_background = cv2.morphologyEx(foreground_logical_matrix, cv2.MORPH_DILATE,
                                              cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8)),
                                              iterations=3)
        dilated_background = 1 - dilated_background

        return foreground_logical_matrix, background_logical_matrix, eroded_foreground, dilated_background

    def load_background_image(self):
        background_image = plt.imread(project_constants.BACKGROUND_IMAGE_PATH)
        background_image = cv2.resize(background_image, (self.frame_width, self.frame_height))
        background_image = cv2.cvtColor(background_image, cv2.COLOR_RGB2BGR)
        return background_image

    @staticmethod
    def create_distance_map_from_probability_maps(binary_frame_fg, binary_frame_bg,
                                                  normalized_foreground_probability_map,
                                                  normalized_background_probability_map):

        foreground_distance_map = GeodisTK.geodesic2d_raster_scan(
            normalized_foreground_probability_map.astype(np.float32), binary_frame_fg, 1, 1)
        background_distance_map = GeodisTK.geodesic2d_raster_scan(
            normalized_background_probability_map.astype(np.float32), binary_frame_bg, 1, 1)

        return foreground_distance_map, background_distance_map

    def create_alpha_frame_from_trimap(self, trimap, trimap_mask, Wf, Wb, foreground_distance_map,
                                       background_distance_map, bound_rect=None):

        alpha_bounding_rect = trimap.copy()
        alpha_bounding_rect[trimap_mask] = Wf[trimap_mask] / (Wf[trimap_mask] + Wb[trimap_mask])
        alpha_bounding_rect[foreground_distance_map == 0] = 1
        alpha_bounding_rect[background_distance_map == 0] = 0
        alpha = np.zeros((self.frame_height, self.frame_width))
        alpha = project_utils.insert_submatrix_from_bounding_rect(alpha, bound_rect,
                                                                  alpha_bounding_rect)
        alpha = cv2.merge([alpha, alpha, alpha])
        return alpha

    @staticmethod
    def create_probability_map(value_channel,
                               P_F_given_c, P_B_given_c, full_frame_value, full_frame_background_logical_mask,
                               full_frame_foreground_logical_mask):

        x_grid = np.linspace(0, 255, 256)
        if P_F_given_c is None and P_B_given_c is None:
            kde_foreground = gaussian_kde(full_frame_value[np.where(full_frame_foreground_logical_mask == 1)],
                                          bw_method='silverman')
            kde_foreground_pdf = kde_foreground.evaluate(x_grid)

            kde_bg = gaussian_kde(full_frame_value[np.where(full_frame_background_logical_mask == 1)],
                                  bw_method='silverman')
            kde_bg_pdf = kde_bg.evaluate(x_grid)

            P_F_given_c = kde_foreground_pdf / (kde_foreground_pdf + kde_bg_pdf)
            P_B_given_c = kde_bg_pdf / (kde_foreground_pdf + kde_bg_pdf)

        foreground_probability_map = P_F_given_c[value_channel]
        background_probability_map = P_B_given_c[value_channel]

        normalized_foreground_probability_map = project_utils.normalize_frame(foreground_probability_map)
        normalized_background_probability_map = project_utils.normalize_frame(background_probability_map)

        return normalized_foreground_probability_map, normalized_background_probability_map, P_F_given_c, P_B_given_c

    @staticmethod
    def create_narrow_band(Vf):
        foreground_mask_eroded = cv2.morphologyEx(Vf, cv2.MORPH_ERODE,
                                                  cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
        narrow_band = (255 * (np.abs(foreground_mask_eroded - Vf) > 0)).astype(np.uint8)
        narrow_band = cv2.morphologyEx(narrow_band, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        return narrow_band

    @staticmethod
    def create_sure_foreground_and_sure_background(foreground_distance_map, background_distance_map, current_shape):
        sure_foreground = np.zeros(current_shape)
        sure_background = np.zeros(current_shape)
        sure_foreground[(foreground_distance_map - background_distance_map) <= 0] = 255
        sure_background[(background_distance_map - foreground_distance_map) <= 0] = 255
        return sure_foreground, sure_background

    def create_trimap(self, foreground_distance_map, background_distance_map):

        current_shape = foreground_distance_map.shape
        Vf, Vb = self.create_sure_foreground_and_sure_background(foreground_distance_map, background_distance_map, current_shape)

        narrow_band = self.create_narrow_band(Vf)
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

    def create_matted_and_alpha_video(self):

        P_F_given_c, P_B_given_c = None, None

        # loop for creating following frames for matted vid #
        for i in range(self.number_of_frames):
            _, full_size_extracted_frame = self.extracted_video_cap.read()
            if full_size_extracted_frame is None:
                break
            else:
                extracted_frame = cv2.resize(full_size_extracted_frame, (self.frame_width, self.frame_height))

            _, full_binary_frame = self.binary_video_cap.read()
            binary_frame = cv2.resize(full_binary_frame, (self.frame_width, self.frame_height))
            _, binary_frame = cv2.threshold(binary_frame[:, :, 0], 0, 255, cv2.THRESH_OTSU)
            _, full_binary_frame = cv2.threshold(full_binary_frame[:, :, 0], 0, 255, cv2.THRESH_OTSU)

            _, full_size_stabilized_frame = self.stabilized_video_cap.read()
            _, _, full_size_value_channel = cv2.split(cv2.cvtColor(full_size_stabilized_frame, cv2.COLOR_BGR2HSV))

            bound_rect = cv2.boundingRect(binary_frame)

            binary_frame = project_utils.slice_frame_from_bounding_rect(binary_frame, bound_rect)
            if not binary_frame.shape[0]:
                continue

            _, _, full_frame_value = cv2.split(cv2.cvtColor(extracted_frame, cv2.COLOR_BGR2HSV))
            value_channel = project_utils.slice_frame_from_bounding_rect(full_frame_value, bound_rect)

            full_frame_foreground_logical_matrix, full_frame_background_logical_matrix, eroded_foreground, \
            dilated_background = self.create_foreground_background_pixels_map(full_binary_frame)

            normalized_foreground_probability_map, normalized_background_probability_map, P_F_given_c, P_B_given_c = \
                self.create_probability_map(
                    value_channel, P_F_given_c,
                    P_B_given_c,
                    full_size_value_channel,
                    dilated_background,
                    eroded_foreground)

            binary_frame_fg = cv2.morphologyEx(binary_frame, cv2.MORPH_ERODE,
                                               cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5)).T, iterations=2)
            binary_frame_bg = cv2.morphologyEx(binary_frame, cv2.MORPH_DILATE,
                                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=6)
            binary_frame_bg = np.bitwise_not(binary_frame_bg)

            foreground_distance_map, background_distance_map = \
                self.create_distance_map_from_probability_maps(binary_frame_fg, binary_frame_bg,
                                                               normalized_foreground_probability_map,
                                                               normalized_background_probability_map)

            trimap = self.create_trimap(foreground_distance_map, background_distance_map)

            trimap_mask = ((trimap == 0.5) & (foreground_distance_map != 0) & (background_distance_map != 0))

            current_size = foreground_distance_map.shape
            Wf, Wb = self.create_Wf_Wb(trimap_mask, foreground_distance_map,
                                                                                 background_distance_map,
                                                                                 normalized_foreground_probability_map,
                                                                                 normalized_background_probability_map,
                                                                                 current_size)

            alpha = self.create_alpha_frame_from_trimap(trimap, trimap_mask, Wf, Wb, foreground_distance_map,
                                                        background_distance_map, bound_rect)

            alpha_original_size = cv2.resize((project_utils.normalize_frame(alpha)).astype(np.uint8),
                                             (self.output_frame_width, self.output_frame_height))

            matted_frame = alpha * extracted_frame + (1 - alpha) * self.background_image

            matted_frame_original_size = cv2.resize(matted_frame, (self.output_frame_width, self.output_frame_height))

            self.matted_video_writer.write(matted_frame_original_size.astype(np.uint8))

            self.alpha_video_writer.write(alpha_original_size)

            self.progress_bar.update(1)

    def close_all_videos(self):

        self.extracted_video_cap.release()
        self.matted_video_writer.release()
        self.alpha_video_writer.release()
        self.binary_video_cap.release()
        self.stabilized_video_cap.release()
        cv2.destroyAllWindows()
        self.progress_bar.close()

    def main_image_matting_module(self):

        self.create_video_writers()

        # P_F_given_c, P_B_given_c = self.handle_first_frame()

        # self.create_matted_and_alpha_video(P_F_given_c, P_B_given_c)

        self.create_matted_and_alpha_video()

        self.close_all_videos()


if __name__ == '__main__':
    # matting_module()
    mat = image_matting()
    mat.main_image_matting_module()
