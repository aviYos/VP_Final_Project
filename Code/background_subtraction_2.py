import cv2
import numpy as np
import project_constants
import project_utils
import matplotlib.pyplot as plt


DIST_2_THRESHOLD = 300
KNN_HISTORY = 20


class background_subtractor:

    def __init__(self):
        self.knn_subtractor = cv2.createBackgroundSubtractorKNN(KNN_HISTORY, DIST_2_THRESHOLD, detectShadows=False)
        self.video_cap = cv2.VideoCapture(project_constants.STABILIZE_PATH)
        self.logger = project_utils.create_general_logger()
        self.all_frames_foreground_mask = None
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
        try:
            median_of_first_k_frames_S = np.median(
                np.array(self.all_frames_Sat_channel_values[0:self.median_kernel_dense]),
                axis=0)
            median_of_last_k_frames_S = np.median(np.array(
                self.all_frames_Sat_channel_values[
                len(self.all_frames_Sat_channel_values) - self.median_kernel_dense:]),
                axis=0)
            return median_of_first_k_frames_S, median_of_last_k_frames_S
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    @staticmethod
    def get_bounding_rect_pixels(frame, bound_rect):
        mask = np.zeros(frame.shape[0:2])
        mask[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]] = 1
        return mask.astype(np.uint8)

    def create_median_mask_for_frame(self, frame_index, median_of_first_k_frames_S, median_of_last_k_frames_S):
        try:
            if frame_index < self.median_kernel_radius:
                median_mask = (self.all_frames_Sat_channel_values[frame_index, :,
                               :] - median_of_first_k_frames_S) > project_constants.MEDIAN_FILTER_THRESHOLD

            elif frame_index > (self.number_of_frames - self.median_kernel_radius - 1):
                median_mask = (self.all_frames_Sat_channel_values[frame_index, :,
                               :] - median_of_last_k_frames_S) > project_constants.MEDIAN_FILTER_THRESHOLD

            else:
                median_per_frame_num_S = np.median(
                    np.array(self.all_frames_Sat_channel_values[frame_index, :, :][
                             frame_index - self.median_kernel_radius:frame_index + self.median_kernel_radius + 1]),
                    axis=0)
                median_mask = ((self.all_frames_Sat_channel_values[frame_index, :,
                                :] - median_per_frame_num_S) > project_constants.MEDIAN_FILTER_THRESHOLD)

            median_mask = np.uint8(255. * median_mask)
            median_mask = cv2.morphologyEx(median_mask, cv2.MORPH_DILATE,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                           iterations=1)
            #plt.imshow(median_mask, cmap='gray')
            #plt.show()

            median_mask, _, _ = self.get_largest_connected_shape_in_mask(
                median_mask)

            #plt.imshow(median_mask, cmap='gray')
            #plt.show()

            return median_mask
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def create_knn_subtractor_mask_for_frame(self, foreground_mask):
        try:
            _, knn_mask = cv2.threshold(foreground_mask, 0, 255, cv2.THRESH_OTSU)  # Binarize the BS frame

            #plt.imshow(knn_mask, cmap='gray')
            #plt.show()

            knn_mask = cv2.morphologyEx(knn_mask, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                        iterations=4)

            #plt.imshow(knn_mask, cmap='gray')
            #plt.show()

            knn_mask = cv2.morphologyEx(knn_mask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                            iterations=7)  # separate foreground from BG noises not filtered by KNN

            #plt.imshow(knn_mask, cmap='gray')
            #plt.show()

            knn_mask, _, _ = self.get_largest_connected_shape_in_mask(
            knn_mask)  # Neutralize noises by eliminating all blobs other than the largest blob

            #plt.imshow(knn_mask, cmap='gray')
            #plt.show()

            # dilate to compensate for erode operations
            return knn_mask
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def create_skin_mask_for_frame(self, hue_channel, sat_channel, value_channel):
        try:
            # Detect skin using thresholding
            skin_mask = (hue_channel < project_constants.SKIN_HUE_threshold) & (
                    sat_channel > project_constants.SKIN_SAT_THRESHOLD_LOW) & (
                                sat_channel < project_constants.SKIN_SAT_THRESHOLD_UPPER) & (
                                value_channel < project_constants.SKIN_VALUE_threshold)
            return np.uint8(255 * skin_mask)
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def union_masks(self, median_filter_mask, knn_mask, Hue, Value):
        try:
            # Unite all masks
            masks_union = median_filter_mask | knn_mask

            #plt.imshow(masks_union, cmap='gray')
            #plt.show()

            masks_union = cv2.morphologyEx(masks_union, cv2.MORPH_DILATE,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                           iterations=2)

            #plt.imshow(masks_union, cmap='gray')
            #plt.show()

            masks_union, _, _ = self.get_largest_connected_shape_in_mask(
                masks_union)  # Neutralize noises by eliminating all blobs other than the largest blob

            # plt.imshow(masks_union, cmap='gray')
            #plt.show()

            # Some processing to reduce noise
            masks_union[Value > project_constants.VALUE_NOISE_THRESHOLD] = 0
            masks_union = cv2.morphologyEx(masks_union, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                           iterations=4)  # close holes in hands

            #plt.imshow(masks_union, cmap='gray')
            #plt.show()

            masks_union[self.frame_height // 2:, :] = cv2.morphologyEx(masks_union[self.frame_height // 2:, :],
                                                                       cv2.MORPH_OPEN,
                                                                       cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                                                 (5, 5)),
                                                                       iterations=3)  # separate legs
            return masks_union

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def get_largest_connected_shape_in_mask(self, mask):
        try:
            number_of_labels, component_labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
            sizes = stats[range(number_of_labels), cv2.CC_STAT_AREA]
            sizes[np.argmax(sizes)] = -1
            largest_shape = np.argmax(sizes)
            mask[component_labels != largest_shape] = 0
            rect_indices = np.array([stats[largest_shape, cv2.CC_STAT_LEFT],  # x - left most(col)
                                     stats[largest_shape, cv2.CC_STAT_TOP],  # y - top most (row)
                                     stats[largest_shape, cv2.CC_STAT_WIDTH],  # WIDTH - the col number for the image
                                     stats[largest_shape, cv2.CC_STAT_HEIGHT]])  # the height number(rows) for the image

            shape_center = (int(centroids[largest_shape, 0]), int(centroids[largest_shape, 1]))
            return mask, rect_indices, shape_center

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    """def largest_component_mask(self, bin_img):
        Finds the largest component in a binary image and returns the component as a mask.

        contours = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        # should be [1] if OpenCV 3+

        max_area = 0
        max_contour_index = 0
        for i, contour in enumerate(contours):
            contour_area = cv2.moments(contour)['m00']
            if contour_area > max_area:
                max_area = contour_area
                max_contour_index = i

        labeled_img = np.zeros(bin_img.shape, dtype=np.uint8)
        cv2.drawContours(labeled_img, contours, max_contour_index, color=255, thickness=-1)

        return labeled_img"""

    def train_background_subtractor_knn(self):
        try:
            self.logger.debug(' training knn subtractor on  stabilized video')

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

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)



    def color_filter(self, frame_hsv, rect, bounding_box_mask, union_masks):

        Hue, Sat, Value = cv2.split(frame_hsv)

        sliced_sat = project_utils.slice_frame_from_bounding_rect(Sat, rect)
        sliced_hue = project_utils.slice_frame_from_bounding_rect(Hue, rect)
        sliced_value = project_utils.slice_frame_from_bounding_rect(Value, rect)

        median_of_sat = np.median(sliced_sat)
        _, out_sat = cv2.threshold(sliced_sat, median_of_sat, 255, cv2.THRESH_BINARY)

        #plt.imshow(out_sat, cmap='gray')
        #plt.show()


        median_of_hue = np.median(sliced_hue)
        _, out_hue = cv2.threshold(sliced_hue, median_of_hue, 255, cv2.THRESH_BINARY)

        #plt.imshow(out_hue, cmap='gray')
        #plt.show()

        median_of_value = np.median(sliced_value)
        _, out_value = cv2.threshold(sliced_value, median_of_value, 255, cv2.THRESH_BINARY)

        #plt.imshow(out_value, cmap='gray')
        #plt.show()

        sliced_union_mask = project_utils.slice_frame_from_bounding_rect(union_masks, rect)
        final_mask = sliced_union_mask & out_sat

        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_DILATE,
                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                      iterations=3)

        final_mask,_,_ = self.get_largest_connected_shape_in_mask(final_mask)
        original_size_final_mask = np.zeros(union_masks.shape)
        original_size_final_mask = project_utils.insert_submatrix_from_bounding_rect(original_size_final_mask, rect, final_mask)
        return original_size_final_mask.astype(np.uint8)

    def main_background_subtraction_module(self):

        self.logger.debug('running background subtraction on stabilized video')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.logger.debug('creating binary video writer handle')

        vid_writer_binary = cv2.VideoWriter(project_constants.BINARY_PATH, fourcc, self.fps,
                                            (self.frame_width, self.frame_height), isColor=False)

        self.logger.debug('creating extracted video writer handle')

        vid_writer_extracted = cv2.VideoWriter(project_constants.EXTRACTED_PATH, fourcc, self.fps,
                                               (self.frame_width, self.frame_height))

        try:

            self.train_background_subtractor_knn()

            self.logger.debug('fetching data for median filter')

            median_of_first_k_frames_S, median_of_last_k_frames_S = self.initialize_median_filter_parameters()

            self.logger.debug('looping over all video frames and running background subtraction')

            for frame_index in range(self.number_of_frames):

                self.logger.debug('running background subtraction on frame number ' + str(frame_index))

                _, frame = self.video_cap.read()
                if frame is None:
                    break

                # B, G, R = cv2.split(frame)
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                self.logger.debug('Transforming frame number ' + str(frame_index) + ' to HSV')

                Hue, Sat, Value = cv2.split(frame_hsv)

                self.logger.debug('Creating median mask for frame number ' + str(frame_index))

                median_filter_mask = self.create_median_mask_for_frame(frame_index, median_of_first_k_frames_S,
                                                                       median_of_last_k_frames_S)

                self.logger.debug('Creating foreground mask for frame number ' + str(frame_index))

                foreground_mask_knn = self.knn_subtractor.apply(Sat)
                knn_mask = self.create_knn_subtractor_mask_for_frame(foreground_mask_knn)

                self.logger.debug('Creating skin mask for frame number ' + str(frame_index))

                self.logger.debug('Creating union mask from all masks for frame number ' + str(frame_index))

                union_masks = self.union_masks(median_filter_mask, knn_mask, Hue, Value)

                self.logger.debug('writing  frame number ' + str(frame_index) + ' to binary video')

                bound_rect = cv2.boundingRect(union_masks)

                bounding_box_mask = self.get_bounding_rect_pixels(frame, bound_rect)

                final_mask = self.color_filter(frame_hsv, bound_rect, bounding_box_mask, union_masks)

                vid_writer_binary.write(final_mask)

                self.logger.debug('writing  frame number ' + str(frame_index) + ' to extracted video')

                vid_writer_extracted.write(cv2.bitwise_and(frame, frame, mask=final_mask.astype(np.uint8)))

            self.logger.debug('saving binary video')

            vid_writer_binary.release()

            self.logger.debug('saving extracted video')

            vid_writer_extracted.release()

            self.logger.debug('closing stabilized video')

            self.video_cap.release()

            cv2.destroyAllWindows()

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)
        finally:
            if 'vid_writer_binary' in locals() and vid_writer_binary.isOpened():
                vid_writer_binary.release()
            if 'vid_writer_extracted' in locals() and vid_writer_extracted.isOpened():
                vid_writer_extracted.release()
            if self.video_cap.isOpened():
                self.video_cap.release()


if __name__ == "__main__":
    background_subtractor_handle = background_subtractor()
    background_subtractor_handle.main_background_subtraction_module()
