import math
from collections import namedtuple
import cv2
import numpy
from scipy import ndimage
from digit_recognizer import DigitRecognizer
from line_detector import find_line, is_point_below_line, get_distance_from_line

digit_recognizer = DigitRecognizer()

Point = namedtuple('Point', ['x', 'y'])
FrameObject = namedtuple('FrameObject', ['center', 'length', 'data'])
FrameResults = namedtuple('FrameResults', ['sum', 'digits'])


def prepare_frame(frame):
    """Prepares frame for processing by converting it to grayscale, thresholding it and removing noise by dilatation."""
    greyscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return_value, threshold_frame = cv2.threshold(greyscale_frame, 167, 255, 0)
    kernel = numpy.ones((2, 2), numpy.uint8)
    prepared_frame = cv2.dilate(threshold_frame, kernel, iterations=2)
    return prepared_frame


def find_objects(frame):
    """Finds object contours on frame."""
    labels, label_cnt = ndimage.label(frame)
    objects = ndimage.find_objects(labels)
    return objects


def crop_object(frame, frame_object):
    """Crops only desired object from whole frame."""
    center_x = (frame_object[1].stop + frame_object[1].start) / 2
    center_y = (frame_object[0].stop + frame_object[0].start) / 2
    center = Point(x=center_x, y=center_y)

    length_x = frame_object[1].stop - frame_object[1].start
    length_y = frame_object[0].stop - frame_object[0].start
    length = Point(x=length_x, y=length_y)

    frame_data = frame[int(center_y-length_y/2):int(center_y+length_y/2),
                       int(center_x-length_x/2):int(center_x+length_x/2)]

    cropped_object = FrameObject(center=center, length=length, data=frame_data)
    return cropped_object


def prepare_object(original_object):
    """Prepares object to be suitable for digit recognition."""
    resized_object = cv2.resize(original_object, (28, 28))
    cleaned_object = cv2.erode(resized_object, numpy.ones((4, 4), numpy.uint8))
    reshaped_object = cleaned_object.reshape(1, 1, 28, 28).astype('float32')
    prepared_object = reshaped_object / 255
    return prepared_object


def is_object_for_recognition(frame_object, min_size):
    """Checks if object is above minimal size in order to remove some noise."""
    return frame_object.length.x > min_size or frame_object.length.y > min_size


def is_object_crossed_line(line, crossing_object):
    """Checks if object is in area below the line which indicates that it has crossed it."""
    if not is_point_below_line(line, crossing_object.center):
        return False

    # consider area around the line as well for cases in which object crossed line with just part of it
    epsilon_x = 0.5 * crossing_object.length.x
    epsilon_y = 0.5 * crossing_object.length.y

    if (line.x1 - epsilon_x <= crossing_object.center.x <= line.x2 + epsilon_x and
            line.y2 - epsilon_y <= crossing_object.center.y <= line.y1 + epsilon_y):
        # ensure that whole object has crossed the line in order to be able to recognize it properly
        distance = get_distance_from_line(line, crossing_object.center)
        diagonal = math.sqrt(pow(crossing_object.length.x, 2) + pow(crossing_object.length.y, 2))
        expected_distance = diagonal / 2
        return 0.895 * expected_distance <= distance <= 1.1 * expected_distance
    return False


def find_sum_of_digits_which_crossed_line(frame, previous_frame_digits):
    """Finds sum of digits which crossed line in currently processed frame."""
    crossed_sum = 0
    line = find_line(frame)
    prepared_frame = prepare_frame(frame)

    objects = find_objects(prepared_frame)

    current_frame_digits = set()

    for frame_object in objects:
        cropped_object = crop_object(prepared_frame, frame_object)

        if not is_object_for_recognition(cropped_object, 10):
            continue

        if not is_object_crossed_line(line, cropped_object):
            continue

        prepared_object = prepare_object(cropped_object.data)
        digit = digit_recognizer.recognize_digit(prepared_object)

        current_frame_digits.add(digit)

        if digit not in previous_frame_digits:
            crossed_sum += digit

    return FrameResults(sum=crossed_sum, digits=current_frame_digits)
