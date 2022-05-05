import cv2
import numpy as np
import project_constants
import project_utils

DIST_2_THRESHOLD = 300
KNN_HISTORY = 10


class background_subtractor_knn:

    def __init__(self):
        self.knn_subtractor = cv2.createBackgroundSubtractorKNN(KNN_HISTORY, DIST_2_THRESHOLD, detectShadows=False)
        self.video_cap = cv2.VideoCapture(project_constants.INPUT_VIDEO_PATH)
        self.logger = None
        self.all_frames_foreground_mask = None
        self.all_frames_foreground_extracted = None
        self.all_frames_binary_mask = None
        self.all_frames_Sat_channel_values = None
        self.number_of_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.kernel = None
        self.fps = project_utils.get_video_fps(self.video_cap)
        self.median_kernel_dense = project_constants.MEDIAN_FILTER_DENSE
        self.median_kernel_radius = (self.median_kernel_dense - 1) // 2

    def initialize_median_filter_parameters(self):
        median_of_first_k_frames_S = np.median(np.array(self.all_frames_Sat_channel_values[0:self.median_kernel_dense]),
                                               axis=0)
        median_of_last_k_frames_S = np.median(np.array(
            self.all_frames_Sat_channel_values[len(self.all_frames_Sat_channel_values) - self.median_kernel_dense:]),
            axis=0)
        return median_of_first_k_frames_S, median_of_last_k_frames_S

    def create_median_mask_for_frame(self, frame_index, median_of_first_k_frames_S, median_of_last_k_frames_S):

        if frame_index < self.median_kernel_radius:
            median_mask = (self.all_frames_Sat_channel_values[frame_index, :,
                           :] - median_of_first_k_frames_S) > project_constants.MEDIAN_FILTER_THRESHOLD

        elif frame_index > (self.number_of_frames - self.median_kernel_radius - 1):
            median_mask = (self.all_frames_Sat_channel_values[frame_index, :,
                           :] - median_of_last_k_frames_S) > project_constants.MEDIAN_FILTER_THRESHOLD

        else:
            median_per_frame_num_S = np.median(
                np.array(self.all_frames_Sat_channel_values[frame_index, :, :][
                         frame_index - self.median_kernel_radius:frame_index + self.median_kernel_radius + 1]), axis=0)
            median_mask = ((self.all_frames_Sat_channel_values[frame_index, :,
                            :] - median_per_frame_num_S) > project_constants.MEDIAN_FILTER_THRESHOLD)

        median_mask = np.uint8(255. * median_mask)
        median_mask = cv2.morphologyEx(median_mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                       iterations=1)
        median_mask, _, _ = self.save_largest_blob(
            median_mask)
        return median_mask

    def create_knn_subtractor_mask_for_frame(self, foreground_mask):
        _, knn_mask = cv2.threshold(foreground_mask, 0, 255, cv2.THRESH_OTSU)  # Binarize the BS frame
        knn_mask = cv2.morphologyEx(knn_mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                    iterations=4)  # separate foreground from BG noises not filtered by KNN
        knn_mask, _, _ = self.save_largest_blob(
            knn_mask)  # Neutralize noises by eliminating all blobs other than the largest blob
        knn_mask = cv2.morphologyEx(knn_mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                    iterations=4)  # dilate to compensate for erode operations

        return knn_mask

    @staticmethod
    def create_skin_mask_for_frame(hue_channel, sat_channel, value_channel):
        # Detect skin using thresholding
        skin_mask = (hue_channel < project_constants.SKIN_HUE_threshold) & (
                sat_channel > project_constants.SKIN_SAT_THRESHOLD_LOW) & (
                            sat_channel < project_constants.SKIN_SAT_THRESHOLD_UPPER) & (
                            value_channel < project_constants.SKIN_VALUE_threshold)
        return np.uint8(255 * skin_mask)

    def union_masks(self, median_filter_mask, skin_mask, knn_mask, Hue, Value):
        # Unite all masks
        masks_union = skin_mask | median_filter_mask | knn_mask
        masks_union = cv2.morphologyEx(masks_union, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                       iterations=2)

        masks_union, _, _ = self.save_largest_blob(
            masks_union)  # Neutralize noises by eliminating all blobs other than the largest blob

        # Some processing to reduce noise
        masks_union[Value > project_constants.VALUE_NOISE_THRESHOLD] = 0
        masks_union = cv2.morphologyEx(masks_union, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                       iterations=4)  # close holes in hands
        masks_union[self.frame_height // 2:, :] = cv2.morphologyEx(masks_union[self.frame_height // 2:, :], cv2.MORPH_OPEN,
                                                     cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                                     iterations=3)  # separate legs
        return masks_union

    @staticmethod
    def save_largest_blob(mask):
        C = mask.copy()
        num_of_labels, component_labels, stats, centroids = cv2.connectedComponentsWithStats(C)
        sizes = stats[range(num_of_labels), cv2.CC_STAT_AREA]
        max_ind = np.argmax(sizes)
        sizes[max_ind] = -1  # neutralize background label so foreground is largest
        largest_blob = np.argmax(sizes)
        C[component_labels != largest_blob] = 0
        rect_indices = np.array([stats[largest_blob, cv2.CC_STAT_LEFT],  # x - left most(col)
                                 stats[largest_blob, cv2.CC_STAT_TOP],  # y - top most (row)
                                 stats[largest_blob, cv2.CC_STAT_WIDTH],  # WIDTH - the col number for the image
                                 stats[largest_blob, cv2.CC_STAT_HEIGHT]])  # the height number(rows) for the image

        blob_center = (int(centroids[largest_blob, 0]), int(centroids[largest_blob, 1]))
        return C, rect_indices, blob_center

    def train_background_subtractor_knn(self):
        self.all_frames_Sat_channel_values = \
            np.zeros((self.number_of_frames, int(self.frame_height), int(self.frame_width)))

        for frame_index in range(self.number_of_frames):
            _, frame = self.video_cap.read()
            if frame is None:
                break

            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            _, sat_channel, _ = cv2.split(frame_hsv)
            _ = self.knn_subtractor.apply(sat_channel)
            self.all_frames_Sat_channel_values[frame_index, :, :] = sat_channel

        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def main_background_subtraction_module(self):

        self.train_background_subtractor_knn()

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        vid_writer_binary = cv2.VideoWriter(project_constants.BINARY_PATH, fourcc, self.fps,
                                            (self.frame_width, self.frame_height), isColor=False)
        vid_writer_extracted = cv2.VideoWriter(project_constants.EXTRACTED_PATH, fourcc, self.fps,
                                               (self.frame_width, self.frame_height))

        median_of_first_k_frames_S, median_of_last_k_frames_S = self.initialize_median_filter_parameters()

        for frame_index in range(self.number_of_frames):

            _, frame = self.video_cap.read()
            if frame is None:
                break

            # B, G, R = cv2.split(frame)
            frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            Hue, Sat, Value = cv2.split(frame_hsv)

            median_filter_mask = self.create_median_mask_for_frame(frame_index, median_of_first_k_frames_S, median_of_last_k_frames_S)

            foreground_mask_knn = self.knn_subtractor.apply(Sat)
            knn_mask = self.create_knn_subtractor_mask_for_frame(foreground_mask_knn)

            skin_mask = self.create_skin_mask_for_frame(Hue, Sat, Value)

            union_masks = self.union_masks(median_filter_mask, skin_mask, knn_mask, Hue, Value)

            vid_writer_binary.write(union_masks)
            vid_writer_extracted.write(cv2.bitwise_and(frame, frame, mask=union_masks))

        vid_writer_binary.release()
        vid_writer_extracted.release()
        self.video_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    background_subtractor_handle = background_subtractor_knn()
    background_subtractor_handle.main_background_subtraction_module()
