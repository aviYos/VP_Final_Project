import logging
import project_constants
import cv2
import json
import numpy as np

ID1 = 315488171
ID2 = 314756297


# function that creates dictionary for run times
def create_time_dictionary():
    time_dictionary = {}.fromkeys(['time_to_stabilize', 'time_to_binary', 'time_to_alpha', 'time_to_matted',
                                   'time_to_output'], None)
    return time_dictionary


# function that creates general logger for debugging
def create_general_logger():
    # Gets or creates a general logger
    log_format = '%(asctime)s : %(levelname)s : %(funcName)s : %(message)s'
    logging.basicConfig(filename=project_constants.GER_NAME, level=logging.DEBUG, format=log_format)
    logger = logging.getLogger(project_constants.GER_NAME)
    return logger


# function that return's video fps
def get_video_fps(video_cap_hanlde):
    return int(video_cap_hanlde.get(cv2.CAP_PROP_FPS))


# function that slices a frame from bouding rect values
def slice_frame_from_bounding_rect(frame, bound_rect):
    return frame[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]]


# function that adds a submatrix define by bounding rect values to bigger image
def insert_submatrix_from_bounding_rect(big, bound_rect, small):
    big[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]] = small
    return big


# function that writes stack of frames to video
def write_frames_to_video(video_full_path, video_fps, frames_to_save, frame_size):
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_Writer_handle = cv2.VideoWriter(video_full_path, fourcc, int(video_fps),
                                              (int(frame_size[0]), int(frame_size[1])))
        for frame in frames_to_save:
            video_Writer_handle.write(frame)
        video_Writer_handle.release()
    except Exception as e:
        print(' counld not save video to path ' + str(video_full_path) + ' Error : ' + str(e))


# function that writes dictionary to json file
def write_to_json_file(file_path: str, dict_to_add: dict):
    with open(file_path, 'w') as f:
        json.dump(dict_to_add, f, indent=4)


# function that saves every frame in video to numpy stack and can convert between color spaces
def load_entire_video(cap, color_space='bgr'):
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for i in range(n_frames):
        success, curr = cap.read()
        if not success:
            break
        if color_space == 'bgr':
            frames.append(curr)
        else:
            frames.append(cv2.cvtColor(curr, cv2.COLOR_BGR2HSV))
        continue
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.asarray(frames)


# function that splits a bounding rect into three parts: top, middle, bottom
def split_bounding_rect(union_masks):
    bounding_rect = cv2.boundingRect(union_masks)
    bounded_mask = slice_frame_from_bounding_rect(union_masks, bounding_rect)

    x, y, w, h = bounding_rect[0], bounding_rect[1], bounding_rect[2], bounding_rect[3]

    # calc top bounding rect
    top_part_rect = (0, 0, w, int(np.floor(h / 3) + 1))

    # calc middle bounding rect
    middle_part_rect = (0, int(np.floor(h / 3)) + 1, w, int(np.floor(h / 3)))

    # calc low bounding rect
    low_part_rect = (0, int(np.floor(2 * h / 3)), w, int(np.floor(h / 3) + 1))

    top_part = slice_frame_from_bounding_rect(bounded_mask, top_part_rect)

    middle_part = slice_frame_from_bounding_rect(bounded_mask, middle_part_rect)

    low_part = slice_frame_from_bounding_rect(bounded_mask, low_part_rect)

    return top_part_rect, middle_part_rect, low_part_rect, top_part, middle_part, low_part, bounded_mask, bounding_rect


def normalize_frame(frame):
    try:
        normalized_frame = (1. * frame - np.amin(frame)) / (np.amax(frame) - np.amin(frame)) * 255
        return normalized_frame
    except Exception as e:
        print(e)
