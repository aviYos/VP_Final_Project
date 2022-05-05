import cv2
import numpy as np
import project_constants




class background_subtractor_mog:

    def __init__(self):
        self.mog_subtractor_handle = cv2.createBackgroundSubtractorMOG2()
        self.video_cap = cv2.VideoCapture(project_constants.SECOND_INPUT_VIDEO_PATH)
        self.logger = None
        self.all_frames_foreground_mask = None
        self.all_frames_binary_mask = None
        self.number_of_frames = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_height = self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_width = self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.kernel = None

    @staticmethod
    def filter_bad_contours(frame, contours):
        mask = np.zeros(frame.shape[:2], dtype=frame.dtype)
        # draw all contours larger than 20 on the mask
        for c in contours:
            if cv2.contourArea(c) > 0:
                x, y, w, h = cv2.boundingRect(c)
                cv2.drawContours(mask, [c], 0, (255), -1)

        # apply the mask to the original image
        frame = cv2.bitwise_and(frame, frame, mask=mask)
        return frame

    def extract_foreground_mask_from_frames(self):
        self.all_frames_foreground_mask = np.zeros((self.number_of_frames, int(self.frame_height), int(self.frame_width)))

        for frame_idx in range(self.number_of_frames):

            ret, frame = self.video_cap.read()
            if frame is None:
                break

            foreground_mask = self.mog_subtractor_handle.apply(frame)
            if not np.any(foreground_mask):
                continue
            else:

                _, foreground_mask = cv2.threshold(foreground_mask, 180, 255, cv2.THRESH_BINARY)
                foreground_mask = cv2.erode(foreground_mask, self.kernel, iterations=1)
                foreground_mask = cv2.dilate(foreground_mask, self.kernel, iterations=2)

                contours, _ = cv2.findContours(foreground_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                foregroundPart = self.filter_bad_contours(frame, contours)
                cv2.imshow('plt',foregroundPart)
                k = cv2.waitKey(1) & 0xff

                if k == ord('q'):
                    break

        self.video_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    background_subtractor_handle = background_subtractor_mog()

    background_subtractor_handle.extract_foreground_mask_from_frames()