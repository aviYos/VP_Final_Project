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
        self.bg_sub_masks = np.zeros((self.number_of_frames, self.frame_height, self.frame_width)).astype(np.uint8)
        self.last_frame = None

    @staticmethod
    def get_bounding_rect_pixels(frame, bound_rect):
        mask = np.zeros(frame.shape[0:2])
        mask[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]] = 1
        return mask.astype(np.uint8)

    def create_knn_subtractor_mask_for_frame(self, foreground_mask):
        try:
            _, knn_mask = cv2.threshold(foreground_mask, 30, 255, cv2.THRESH_OTSU)  # Binarize the BS frame

            knn_mask, _, _ = self.get_largest_connected_shape_in_mask(
                knn_mask)  # Neutralize noises by eliminating all blobs other than the largest bloB

            knn_mask = cv2.morphologyEx(knn_mask, cv2.MORPH_DILATE,
                                        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                        iterations=4)

            knn_mask = cv2.morphologyEx(knn_mask, cv2.MORPH_ERODE,
                                        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
                                        iterations=7)

            knn_mask, _, _ = self.get_largest_connected_shape_in_mask(
                knn_mask)  # Neutralize noises by eliminating all blobs other than the largest bloB

            #plt.imshow(knn_mask, cmap='gray')
            #plt.show()
            return knn_mask
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def union_masks(self, median_filter_mask, knn_mask):
        try:
            # Unite all masks
            masks_union = median_filter_mask | knn_mask

            masks_union, _, _ = self.get_largest_connected_shape_in_mask(
                masks_union)

            # reduce noise
            masks_union = cv2.morphologyEx(masks_union, cv2.MORPH_ERODE,
                                           cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 5)),
                                           iterations=13)  # close holes in hands

            masks_union, _, _ = self.get_largest_connected_shape_in_mask(
                masks_union)

            masks_union = cv2.morphologyEx(masks_union, cv2.MORPH_DILATE,
                                           cv2.getStructuringElement(cv2.MORPH_CROSS, (1, 10)),
                                           iterations=5)  # close holes in hands

            plt.imshow(masks_union, cmap='gray')
            # plt.show()

            masks_union[self.frame_height // 2:, :] = cv2.morphologyEx(masks_union[self.frame_height // 2:, :],
                                                                       cv2.MORPH_OPEN,
                                                                       cv2.getStructuringElement(cv2.MORPH_RECT,
                                                                                                 (5, 5)),
                                                                       iterations=3)  # separate legs

            masks_union = cv2.morphologyEx(masks_union, cv2.MORPH_DILATE,
                                           cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
                                           iterations=2)

            masks_union, _, _ = self.get_largest_connected_shape_in_mask(
                masks_union)

            #plt.imshow(masks_union, cmap='gray')

            return masks_union

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def apply_grabcut(self, bound_rect, frame_hsv, union_mask):
        try:

            eroded_mask = cv2.morphologyEx(union_mask, cv2.MORPH_ERODE,
                                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                           iterations=2)

            dilated_mask = cv2.morphologyEx(union_mask, cv2.MORPH_DILATE,
                                            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
                                            iterations=2)

            diff = dilated_mask - eroded_mask
            _, diff = cv2.threshold(diff, 30, 255, cv2.THRESH_OTSU)  # Binarize the BS frame

            diff2, _, _ = self.get_largest_connected_shape_in_mask(
                diff)

            mask = np.zeros(frame_hsv.shape[:2], np.uint8)
            bgdModel = np.zeros((1, 65), np.float64)
            fgdModel = np.zeros((1, 65), np.float64)
            cv2.grabCut(frame_hsv, mask, bound_rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            frame_hsv = frame_hsv * mask2[:, :, np.newaxis]
            plt.imshow(frame_hsv), plt.colorbar(), plt.show()
        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    @staticmethod
    def find_largest_contour(frame, fill=False):

        frame = frame.astype(np.uint8)
        contours, hierarchy = cv2.findContours(
            frame,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )
        largest_contour = max(contours, key=cv2.contourArea)
        return largest_contour

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

    def train_background_subtractor_knn(self):
        T = 5
        all_hsv_frames = []
        try:
            self.logger.debug(' training knn subtractor on  stabilized video')

            self.all_frames_Sat_channel_values = \
                np.zeros((self.number_of_frames, int(self.frame_height), int(self.frame_width)))

            self.bg_sub_masks = np.zeros((self.number_of_frames, self.frame_height, self.frame_width)).astype(np.uint8)

            for i in range(T):
                for frame_index in range(self.number_of_frames):
                    _, frame = self.video_cap.read()
                    if frame is None:
                        break
                    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    if not i:
                        all_hsv_frames.append(frame_hsv)

                    _, sat_channel, value_channel = cv2.split(frame_hsv)

                    self.all_frames_Sat_channel_values[frame_index, :, :] = sat_channel
                    self.bg_sub_masks[frame_index, :, :] = self.knn_subtractor.apply(sat_channel)
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            hsv_median_frame = np.median(all_hsv_frames, axis=0).astype(dtype=np.uint8)
            return hsv_median_frame

        except Exception as e:
            self.logger.error('Error in background subtraction: ' + str(e), exc_info=True)

    def try_k_means(self, frame_hsv, rect):
        frame = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
        frame = project_utils.slice_frame_from_bounding_rect(frame, rect)
        twoDimage = frame.reshape((-1, 3))
        twoDimage = np.float32(twoDimage)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        attempts = 10
        ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
        center = np.uint8(center)
        res = center[label.flatten()]
        result_image = res.reshape((frame.shape))
        plt.imshow(result_image)

    def get_image_derivative(self, frame):
        edges = cv2.Canny(frame, 100, 200)
        eroded_mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE,
                                       cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)),
                                       iterations=2)
        plt.imshow(eroded_mask, cmap='gray')
        self.find_largest_contour(eroded_mask)

    def color_filter(self, frame_hsv, rect, bounding_box_mask, union_masks):

        bgr = cv2.cvtColor(frame_hsv, cv2.COLOR_HSV2BGR)
        sliced_hsv_image = project_utils.slice_frame_from_bounding_rect(frame_hsv, rect)
        bilateral_filter = cv2.bilateralFilter(sliced_hsv_image, 9, 75, 75)
        self.try_k_means(sliced_hsv_image, rect)

        return union_masks.astype(np.uint8)

    def new_median_filter(self, median_frame_hsv, hsv):
        dframe = cv2.absdiff(hsv, median_frame_hsv)
        _, dframe_sat = cv2.threshold(dframe[:, :, 1], 25, 255, cv2.THRESH_BINARY)

        dframe_sat, _, _ = self.get_largest_connected_shape_in_mask(dframe_sat)

        _, dframe_val = cv2.threshold(dframe[:, :, 2], 40, 255, cv2.THRESH_BINARY)

        dframe_val, _, _ = self.get_largest_connected_shape_in_mask(dframe_val)

        dframe_sat = cv2.morphologyEx(dframe_sat, cv2.MORPH_DILATE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6)),
                                      iterations=2)

        dframe_sat = cv2.morphologyEx(dframe_sat, cv2.MORPH_ERODE,
                                      cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)),
                                      iterations=6)

        # median_mask = dframe_sat & dframe_val

        median_mask = dframe_sat

        final_median_mask, _, _ = self.get_largest_connected_shape_in_mask(median_mask)

        plt.imshow(final_median_mask)

        return final_median_mask

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

                # B, G, R = cv2.split(frame)
                frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                self.logger.debug('Transforming frame number ' + str(frame_index) + ' to HSV')

                Hue, Sat, Value = cv2.split(frame_hsv)

                median_filter_mask = self.new_median_filter(median_frame_hsv, frame_hsv)

                self.logger.debug('Creating median mask for frame number ' + str(frame_index))

                self.logger.debug('Creating foreground mask for frame number ' + str(frame_index))

                foreground_mask_knn = self.bg_sub_masks[frame_index, :, :]
                knn_mask = self.create_knn_subtractor_mask_for_frame(foreground_mask_knn)

                self.logger.debug('Creating skin mask for frame number ' + str(frame_index))

                self.logger.debug('Creating union mask from all masks for frame number ' + str(frame_index))

                final_mask = self.union_masks(median_filter_mask, knn_mask)

                self.logger.debug('writing  frame number ' + str(frame_index) + ' to binary video')

                bound_rect = cv2.boundingRect(final_mask)

                bounding_box_mask = self.get_bounding_rect_pixels(frame, bound_rect)

                # final_mask = self.apply_grabcut(bound_rect,frame, union_masks)

                # final_mask = self.color_filter(frame_hsv, bound_rect, bounding_box_mask, final_mask)

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
