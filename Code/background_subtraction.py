import cv2
import numpy as np
import project_constants
import project_utils


class background_subtractor:

    def __init__(self):
        """background_subtractor Init"""
        self.knn_subtractor = cv2.createBackgroundSubtractorKNN()
        self.video_cap = cv2.VideoCapture(project_constants.STABILIZE_PATH)
        self.logger = project_utils.create_general_logger()
        self.number_of_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.kernel = None
        self.fps = project_utils.get_video_fps(self.video_cap)
        self.bg_sub_masks = np.zeros((self.number_of_frames, self.frame_height, self.frame_width)).astype(np.uint8)

    @staticmethod
    def get_bounding_rect_pixels(frame, bound_rect):
        """get bounding rect pixels logical indexes of a binary mask"""
        mask = np.zeros(frame.shape[0:2])
        mask[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]] = 1
        return mask.astype(np.uint8)

    """ def filter_noise_with_kde(self, binary_frame, full_size_value_channel, P_F_given_c, P_B_given_c, extracted):
    Use kde of the foreground  to filter noise
        _, _, extracted_value = cv2.split(cv2.cvtColor(extracted, cv2.COLOR_BGR2HSV))
        bound_rect = cv2.boundingRect(binary_frame)
        bounded_binary_frame = project_utils.slice_frame_from_bounding_rect(binary_frame, bound_rect)
        value_channel = project_utils.slice_frame_from_bounding_rect(extracted_value, bound_rect)
        _, _, foreground_mask, background_mask = \
            matting_module.create_foreground_background_pixels_map(binary_frame)

        foreground_binary, _, bounded_foreground_mask, bounded_background_mask = \
            matting_module.create_foreground_background_pixels_map(bounded_binary_frame)

        binary_frame_fg = cv2.morphologyEx(foreground_binary, cv2.MORPH_ERODE,
                                           cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5)).T, iterations=2)
        binary_frame_bg = cv2.morphologyEx(foreground_binary, cv2.MORPH_DILATE,
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=6)
        binary_frame_bg = np.bitwise_not(binary_frame_bg)

        normalized_foreground_probability_map, normalized_background_probability_map, P_F_given_c, P_B_given_c = \
            matting_module.create_probability_map(
                value_channel, P_F_given_c,
                P_B_given_c,
                full_size_value_channel,
                background_mask,
                foreground_mask)

        foreground_distance_map, background_distance_map = \
            matting_module.create_distance_map_from_probability_maps(binary_frame_fg, binary_frame_bg,
                                                                     normalized_foreground_probability_map,
                                                                     normalized_background_probability_map)

        return P_F_given_c, P_B_given_c
    """

    def create_knn_subtractor_mask_for_frame(self, foreground_mask):
        """ Applying some noise cleaning tricks on the knn subtractor"""
        try:

            h, w = foreground_mask.shape
            knn_mask = foreground_mask.copy()

            # shadow value is 127 so new label shadow as background
            knn_mask[knn_mask < 255] = 0

            # clear small noises from the frame expect the legs
            knn_mask[:int(np.floor(2 * h / 3)), :] = cv2.morphologyEx(knn_mask[:int(np.floor(2 * h / 3)), :],
                                                                      cv2.MORPH_OPEN,
                                                                      cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                                                                      iterations=1)

            # close holes in the head
            knn_mask[:int(np.floor(2 * h / 5)), :] = cv2.morphologyEx(knn_mask[:int(np.floor(2 * h / 5)), :],
                                                                      cv2.MORPH_CLOSE,
                                                                      cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                                                (1, 10)),
                                                                      iterations=3)

            # close holes in the legs
            knn_mask[int(np.floor(2 * h / 3)):, :] = cv2.morphologyEx(knn_mask[int(np.floor(2 * h / 3)):, :],
                                                                      cv2.MORPH_CLOSE,
                                                                      cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                                (1, 20)),
                                                                      iterations=3)

            # remove small noise shapes
            knn_mask = self.get_largest_connected_shape_in_mask(knn_mask)

            return knn_mask
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def create_final_mask_from_sub_masks(self, original_masks_union, top_bound_rect, middle_bound_rect, down_bound_rect,
                                         top_mask,
                                         middle_mask, down_mask, bounded_mask, bound_rect_mask):
        """ merge the masks of the head, legs and middle body to one mask"""
        try:
            # create new matrices
            final_mask = np.zeros(original_masks_union.shape)
            new_bounded_mask = np.zeros(bounded_mask.shape)

            # insert head mask for new matrix
            new_bounded_mask = project_utils.insert_submatrix_from_bounding_rect(new_bounded_mask, top_bound_rect,
                                                                                 top_mask)
            # insert body mask for new matrix
            new_bounded_mask = project_utils.insert_submatrix_from_bounding_rect(new_bounded_mask, middle_bound_rect,
                                                                                 middle_mask)

            # insert legs mask for new matrix
            new_bounded_mask = project_utils.insert_submatrix_from_bounding_rect(new_bounded_mask, down_bound_rect,
                                                                                 down_mask)
            final_mask = project_utils.insert_submatrix_from_bounding_rect(final_mask, bound_rect_mask,
                                                                           new_bounded_mask)
            return final_mask
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def union_masks(self, median_filter_mask, knn_mask):
        """ merge the knn mask and the median mask to one binary mask"""
        try:
            # merge all masks
            masks_union = median_filter_mask | knn_mask

            # remove small noise shapes
            masks_union = self.get_largest_connected_shape_in_mask(masks_union)

            # split masks to sub masks
            top_bound_rect, middle_bound_rect, down_bound_rect, top_mask, middle_mask, down_mask, \
                bound_rect_mask, bound_rect = project_utils.split_bounding_rect(masks_union)

            # erode and dilate for noise cleaning
            top_mask = cv2.morphologyEx(top_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                        iterations=2)

            middle_mask = cv2.morphologyEx(middle_mask, cv2.MORPH_OPEN,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (2, 9)),
                                           iterations=1)

            middle_mask = cv2.morphologyEx(middle_mask, cv2.MORPH_CLOSE,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                                           iterations=1)

            down_mask = cv2.morphologyEx(down_mask, cv2.MORPH_CLOSE,
                                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6, 6)),
                                         iterations=3)

            # merge sub masks
            final_mask = self.create_final_mask_from_sub_masks(masks_union, top_bound_rect,
                                                               middle_bound_rect, down_bound_rect, top_mask,
                                                               middle_mask, down_mask, bound_rect_mask, bound_rect)
            # remove small noise shapes again
            final_mask = self.get_largest_connected_shape_in_mask(final_mask.astype(np.uint8))

            return final_mask.astype(np.uint8)

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def get_largest_connected_shape_in_mask(self, mask):
        """ return only the biggest mask except the background"""
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
        """ Train the knn subtractor of cv2 T times on the video"""
        T = project_constants.TRAIN_ITER
        all_hsv_frames = []
        try:
            self.logger.debug(' training knn subtractor on stabilized video')

            # init matrix for knn background subtractor results
            self.bg_sub_masks = np.zeros((self.number_of_frames, self.frame_height, self.frame_width)).astype(np.uint8)

            for i in range(T):
                for frame_index in range(self.number_of_frames):
                    _, frame = self.video_cap.read()
                    if frame is None:
                        break

                    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                    # blur frame to remove noise
                    frame = cv2.medianBlur(frame, 3)
                    if not i:
                        all_hsv_frames.append(frame_hsv)

                    # apply knn background subtractor
                    self.bg_sub_masks[frame_index, :, :] = self.knn_subtractor.apply(frame)
                # move video pointer to the beginning
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # return median frame for further use
            hsv_median_frame = np.median(all_hsv_frames, axis=0).astype(dtype=np.uint8)
            return hsv_median_frame

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def median_filter(self, median_frame_hsv, hsv):
        """ Median filter to extract the foreground based on hsv colorspace"""
        try:

            # abs diff to get pixels value and sat difference from median
            dframe = cv2.absdiff(hsv, median_frame_hsv)

            # transform the difference to binary mask
            _, dframe_val = cv2.threshold(dframe[:, :, 1], 40, 255, cv2.THRESH_BINARY)

            # remove small noise shapes
            dframe_val = self.get_largest_connected_shape_in_mask(dframe_val)

            h, w = dframe_val.shape

            # erode and dilate for noise cleaning
            dframe_val[:int(np.floor(h / 2)), :] = cv2.morphologyEx(dframe_val[:int(np.floor(h / 2)), :],
                                                                    cv2.MORPH_OPEN,
                                                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                              (3, 3)),
                                                                    iterations=4)

            dframe_val[int(np.floor(h / 2)):, :] = cv2.morphologyEx(dframe_val[int(np.floor(h / 2)):, :],
                                                                    cv2.MORPH_OPEN,
                                                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                              (3, 3)),
                                                                    iterations=2)

            dframe_val[int(np.floor(h / 2)):, :] = cv2.morphologyEx(dframe_val[int(np.floor(h / 2)):, :],
                                                                    cv2.MORPH_CLOSE,
                                                                    cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                                                              (4, 4)),
                                                                    iterations=3)

            # remove small noise shapes  again
            dframe_val = self.get_largest_connected_shape_in_mask(dframe_val)

            # erode and dilate for noise cleaning
            _, dframe_sat = cv2.threshold(dframe[:, :, 2], 40, 255, cv2.THRESH_BINARY)

            dframe_sat = cv2.morphologyEx(dframe_sat, cv2.MORPH_OPEN,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                          iterations=4)

            dframe_sat = cv2.morphologyEx(dframe_sat, cv2.MORPH_CLOSE,
                                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)),
                                          iterations=4)
            # remove small noise shapes
            dframe_sat = self.get_largest_connected_shape_in_mask(dframe_sat)

            median_mask = np.zeros(dframe_val.shape)
            # merge masks
            median_mask[np.where(dframe_val > 0) or np.where(dframe_sat > 0)] = 255
            median_mask = median_mask.astype(np.uint8)
            # remove small noise shapes
            final_median_mask = self.get_largest_connected_shape_in_mask(median_mask)

            return final_median_mask
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def main_background_subtraction_module(self):
        """ main background subtraction module"""

        self.logger.debug('running background subtraction on stabilized video')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        self.logger.debug('creating binary video writer handle')

        vid_writer_binary = cv2.VideoWriter(project_constants.BINARY_PATH, fourcc, self.fps,
                                            (self.frame_width, self.frame_height), isColor=False)

        self.logger.debug('creating extracted video writer handle')

        vid_writer_extracted = cv2.VideoWriter(project_constants.EXTRACTED_PATH, fourcc, self.fps,
                                               (self.frame_width, self.frame_height))

        try:

            # training the knn background subtractor
            median_frame_hsv = self.train_background_subtractor_knn()

            self.logger.debug('looping over all video frames and running background subtraction')

            for frame_index in range(self.number_of_frames):

                self.logger.debug('running background subtraction on frame number ' + str(frame_index))

                # read frame
                _, frame = self.video_cap.read()
                if frame is None:
                    break

                self.logger.debug('Transforming frame number ' + str(frame_index) + ' to HSV')

                # transform to hsv
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                _, _, value_channel = cv2.split(frame_hsv)

                self.logger.debug('Creating median mask for frame number ' + str(frame_index))

                # create the median filter mask and remove noise
                median_filter_mask = self.median_filter(median_frame_hsv, frame_hsv)

                self.logger.debug('Creating knn foreground mask for frame number ' + str(frame_index))

                # remove noises from the knn mask
                foreground_mask_knn = self.bg_sub_masks[frame_index, :, :]
                knn_mask = self.create_knn_subtractor_mask_for_frame(foreground_mask_knn)

                self.logger.debug('Creating union mask from all masks for frame number ' + str(frame_index))

                # merge the knn and the median mask to one mask

                final_mask = self.union_masks(median_filter_mask, knn_mask)

                extracted = cv2.bitwise_and(frame, frame, mask=final_mask.astype(np.uint8))

                self.logger.debug('writing frame number ' + str(frame_index) + ' to binary video')

                vid_writer_binary.write(final_mask)

                self.logger.debug('writing  frame number ' + str(frame_index) + ' to extracted video')

                vid_writer_extracted.write(extracted)

            # release and save all videos

            self.logger.debug('saving binary video')

            vid_writer_binary.release()

            self.logger.debug('saving extracted video')

            vid_writer_extracted.release()

            self.logger.debug('closing stabilized video')

            self.video_cap.release()

            # close all open windows
            cv2.destroyAllWindows()

        except Exception as e:
            # write error to log and safe quit
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)
        finally:
            # close all video handles
            if 'vid_writer_binary' in locals() and vid_writer_binary.isOpened():
                vid_writer_binary.release()
            if 'vid_writer_extracted' in locals() and vid_writer_extracted.isOpened():
                vid_writer_extracted.release()
            if self.video_cap.isOpened():
                self.video_cap.release()


if __name__ == "__main__":
    background_subtractor_handle = background_subtractor()
    background_subtractor_handle.main_background_subtraction_module()
