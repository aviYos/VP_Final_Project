import cv2
import numpy as np
import project_constants
import project_utils
import matplotlib.pyplot as plt


class background_subtractor:

    def __init__(self):
        self.knn_subtractor = cv2.createBackgroundSubtractorKNN()
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
        self.bg_sub_masks = np.zeros((self.number_of_frames, self.frame_height, self.frame_width)).astype(np.uint8)
        self.last_frame = None

    @staticmethod
    def get_bounding_rect_pixels(frame, bound_rect):
        mask = np.zeros(frame.shape[0:2])
        mask[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]] = 1
        return mask.astype(np.uint8)

    def create_knn_subtractor_mask_for_frame(self, foreground_mask):
        try:
            _, knn_mask = cv2.threshold(foreground_mask, 200, 255, cv2.THRESH_BINARY)  # shadow as background

            h, w = knn_mask.shape

            knn_mask[:int(np.floor(h / 2)), :] = cv2.morphologyEx(knn_mask[:int(np.floor(h / 2)), :], cv2.MORPH_OPEN,
                                                                  cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                                                                  iterations=1)

            knn_mask[:int(np.floor(h / 2)), :] = cv2.morphologyEx(knn_mask[:int(np.floor(h / 2)), :], cv2.MORPH_CLOSE,
                                                                  cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                                                  iterations=5)

            knn_mask[int(np.floor(2 * h / 3)):, :] = cv2.morphologyEx(knn_mask[int(np.floor(2 * h / 3)):, :],
                                                                      cv2.MORPH_CLOSE,
                                                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                                (6, 6)),
                                                                      iterations=1)

            knn_mask = self.get_largest_connected_shape_in_mask(knn_mask)
            return knn_mask
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def create_final_mask_from_sub_masks(self, original_masks_union, top_bound_rect, middle_bound_rect, down_bound_rect,
                                         top_mask,
                                         middle_mask, down_mask, bounded_mask, bound_rect_mask):
        try:
            final_mask = np.zeros(original_masks_union.shape)
            new_bounded_mask = np.zeros(bounded_mask.shape)
            new_bounded_mask = project_utils.insert_submatrix_from_bounding_rect(new_bounded_mask, top_bound_rect,
                                                                                 top_mask)
            new_bounded_mask = project_utils.insert_submatrix_from_bounding_rect(new_bounded_mask, middle_bound_rect,
                                                                                 middle_mask)
            new_bounded_mask = project_utils.insert_submatrix_from_bounding_rect(new_bounded_mask, down_bound_rect,
                                                                                 down_mask)
            final_mask = project_utils.insert_submatrix_from_bounding_rect(final_mask, bound_rect_mask,
                                                                           new_bounded_mask)
            return final_mask
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def union_masks(self, median_filter_mask, knn_mask):
        try:
            # merge all masks
            masks_union = median_filter_mask | knn_mask

            masks_union = self.get_largest_connected_shape_in_mask(
                masks_union)

            top_bound_rect, middle_bound_rect, down_bound_rect, top_mask, middle_mask, down_mask, \
                bound_rect_mask, bound_rect = project_utils.split_bounding_rect(masks_union)

            top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                        iterations=7)

            middle_mask = cv2.morphologyEx(middle_mask, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                                           iterations=1)

            down_mask = cv2.morphologyEx(down_mask, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)),
                                         iterations=3)

            final_mask = self.create_final_mask_from_sub_masks(masks_union, top_bound_rect,
                                                               middle_bound_rect, down_bound_rect, top_mask,
                                                               middle_mask, down_mask, bound_rect_mask, bound_rect)

            return final_mask.astype(np.uint8)

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def get_largest_connected_shape_in_mask(self, mask):
        try:
            number_of_labels, component_labels, stats, _ = cv2.connectedComponentsWithStats(mask)
            sizes = stats[range(number_of_labels), cv2.CC_STAT_AREA]
            # - remove black background part
            sizes[np.argmax(sizes)] = -1
            largest_shape = np.argmax(sizes)
            mask[component_labels != largest_shape] = 0
            return mask

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def train_background_subtractor_knn(self):
        T = 7
        all_hsv_frames = []
        try:
            self.logger.debug(' training knn subtractor on stabilized video')

            self.all_frames_Sat_channel_values = \
                np.zeros((self.number_of_frames, int(self.frame_height), int(self.frame_width)))

            self.bg_sub_masks = np.zeros((self.number_of_frames, self.frame_height, self.frame_width)).astype(np.uint8)

            for i in range(T):
                for frame_index in range(self.number_of_frames):
                    _, frame = self.video_cap.read()
                    if frame is None:
                        break
                    else:
                        frame = cv2.GaussianBlur(frame, (31, 31), cv2.BORDER_REFLECT_101)

                    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    if not i:
                        all_hsv_frames.append(frame_hsv)

                    _, sat_channel, value_channel = cv2.split(frame_hsv)

                    self.all_frames_Sat_channel_values[frame_index, :, :] = sat_channel
                    self.bg_sub_masks[frame_index, :, :] = self.knn_subtractor.apply(frame_hsv[:, :, 1:])
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            hsv_median_frame = np.median(all_hsv_frames, axis=0).astype(dtype=np.uint8)
            return hsv_median_frame

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def new_median_filter(self, median_frame_hsv, hsv):
        try:
            dframe = cv2.absdiff(hsv, median_frame_hsv)

            _, dframe_sat = cv2.threshold(dframe[:, :, 1], 40, 255, cv2.THRESH_BINARY)

            dframe_sat = self.get_largest_connected_shape_in_mask(dframe_sat)

            dframe_sat = cv2.morphologyEx(dframe_sat, cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                          iterations=1)

            dframe_sat = self.get_largest_connected_shape_in_mask(dframe_sat)

            _, dframe_val = cv2.threshold(dframe[:, :, 2], 40, 255, cv2.THRESH_BINARY)

            dframe_val = cv2.morphologyEx(dframe_val, cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
                                          iterations=1)

            dframe_val = self.get_largest_connected_shape_in_mask(dframe_val)

            median_mask = np.zeros(dframe_sat.shape)
            median_mask[np.where(np.logical_and(dframe_sat > 0, dframe_val > 0))] = 255
            median_mask = median_mask.astype(np.uint8)
            final_median_mask = self.get_largest_connected_shape_in_mask(median_mask)

            return final_median_mask
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

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

            median_frame_hsv = self.train_background_subtractor_knn()

            self.logger.debug('looping over all video frames and running background subtraction')

            for frame_index in range(self.number_of_frames):

                self.logger.debug('running background subtraction on frame number ' + str(frame_index))

                _, frame = self.video_cap.read()
                if frame is None:
                    break

                self.logger.debug('Transforming frame number ' + str(frame_index) + ' to HSV')

                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                self.logger.debug('Creating median mask for frame number ' + str(frame_index))

                median_filter_mask = self.new_median_filter(median_frame_hsv, frame_hsv)

                self.logger.debug('Creating knn foreground mask for frame number ' + str(frame_index))

                foreground_mask_knn = self.bg_sub_masks[frame_index, :, :]
                knn_mask = self.create_knn_subtractor_mask_for_frame(foreground_mask_knn)

                self.logger.debug('Creating union mask from all masks for frame number ' + str(frame_index))

                final_mask = self.union_masks(median_filter_mask, knn_mask)

                self.logger.debug('writing frame number ' + str(frame_index) + ' to binary video')

                vid_writer_binary.write(final_mask)

                self.logger.debug('writing  frame number ' + str(frame_index) + ' to extracted video')

                vid_writer_extracted.write(cv2.bitwise_and(frame, frame, mask=final_mask.astype(np.uint8)))

                self.last_frame = frame_hsv

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
