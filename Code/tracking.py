import numpy as np
import cv2
from tqdm import tqdm
from project_utils import write_to_json_file
import project_constants


def get_video_parameters(capture: cv2.VideoCapture) -> dict:
    """Get an OpenCV capture object and extract its parameters.
    Args:
        capture: cv2.VideoCapture object.
    Returns:
        parameters: dict. Video parameters extracted from the video.

    """
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    fps = int(capture.get(cv2.CAP_PROP_FPS))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    return {"fourcc": fourcc, "fps": fps, "height": height, "width": width,
            "frame_count": frame_count}


def tracking(
        matted_video_path: str, binary_video_path, output_video_path: str) -> None:
    print('Tracking video:')
    cap_extracted = cv2.VideoCapture(matted_video_path)
    cap_binary = cv2.VideoCapture(binary_video_path)
    num_of_frames = int(cap_extracted.get(cv2.CAP_PROP_FRAME_COUNT))
    vid_params = get_video_parameters(cap_extracted)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = vid_params.get("fps")
    size = vid_params.get("width"), vid_params.get("height")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, size)
    tracking_dict = {}
    pbar = tqdm(total=num_of_frames)
    for i in range(num_of_frames):
        ret_extracted, extracted_frame = cap_extracted.read()
        if not ret_extracted:
            break
        ret_binary, binary_frame = cap_binary.read()
        if not ret_binary:
            break
        if len(binary_frame.shape) == 3:
            binary_frame = cv2.cvtColor(binary_frame, cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(binary_frame)
        tracking_dict[int(i+1)] = [x, y, h, w]
        cv2.rectangle(extracted_frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
        out.write(np.uint8(extracted_frame))
        pbar.update(1)
        if cv2.waitKey(1) == ord('q'):
            break
    pbar.close()
    cv2.destroyAllWindows()
    cap_extracted.release()
    cap_binary.release()
    out.release()
    write_to_json_file(project_constants.TRACKING_LOGGER, tracking_dict)
