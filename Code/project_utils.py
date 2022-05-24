import logging
import project_constants
import cv2
import json

ID1 = 315488171
ID2 = 314756297


def create_time_dictionary():
    time_dictionary = {}.fromkeys(['time_to_stabilize', 'time_to_binary', 'time_to_alpha', 'time_to_matted',
                                   'time_to_output'], None)
    return time_dictionary


def create_general_logger():
    # Gets or creates a general logger
    log_format = '%(asctime)s : %(levelname)s : %(funcName)s : %(message)s'
    logging.basicConfig(filename=project_constants.GER_NAME, level=logging.DEBUG, format=log_format)
    logger = logging.getLogger(project_constants.GER_NAME)
    return logger


def get_video_fps(video_cap_hanlde):
    return int(video_cap_hanlde.get(cv2.CAP_PROP_FPS))


def slice_frame_from_bounding_rect(frame, bound_rect):
    return frame[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]]


def insert_submatrix_from_bounding_rect(big, bound_rect, small):
     big[bound_rect[1]:bound_rect[1] + bound_rect[3], bound_rect[0]:bound_rect[0] + bound_rect[2]] = small
     return big


def write_frames_to_video(video_full_path, video_fps, frames_to_save, frame_size):
    try:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_Writer_handle = cv2.VideoWriter(video_full_path, fourcc, int(video_fps), (int(frame_size[0]), int(frame_size[1])))
        for frame in frames_to_save:
            video_Writer_handle.write(frame)
        video_Writer_handle.release()
    except Exception as e:
        print(' counld not save video to path ' + str(video_full_path) + ' Error : ' + str(e))


def write_to_json_file(file_path: str, dict_to_add: dict):
    with open(file_path, 'w') as f:
        json.dump(dict_to_add, f, indent=4)
