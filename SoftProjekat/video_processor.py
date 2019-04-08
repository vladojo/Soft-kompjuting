import cv2
from frame_processor import find_sum_of_digits_which_crossed_line


def process_video(file_path):
    """Processes single video and return sum of numbers which crossed the line."""
    video = cv2.VideoCapture(file_path)

    video_sum = 0
    frame_cnt = 0

    previous_frame_digits = set()

    while True:
        return_value, frame = video.read()

        # skip same frames to improve performance
        frame_cnt += 1
        if frame_cnt % 2 != 0:
            continue

        if not return_value:
            break

        frame_results = find_sum_of_digits_which_crossed_line(frame, previous_frame_digits)
        video_sum += frame_results.sum

        previous_frame_digits = frame_results.digits

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()

    return video_sum
