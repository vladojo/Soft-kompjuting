from collections import namedtuple
import math
import cv2
import numpy as np

Line = namedtuple('Line', ['x1', 'y1', 'x2', 'y2', 'k', 'n', 'length'])


def find_line(image):
    """Finds line on the image and calculate its properties.
       Image is converted to binary, and then probabilistic Hough transformation is applied to it to detect the line."""
    greyscale_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return_value, thresholds = cv2.threshold(greyscale_img, 20, 255, cv2.THRESH_BINARY)

    ro = 1
    theta = np.pi / 180
    min_line_length = 150
    max_line_gap = 5
    lines = cv2.HoughLinesP(thresholds, ro, theta, min_line_length, min_line_length, max_line_gap)

    # remember coordinates of line ends
    x1 = min(lines[:, 0, 0])
    x2 = max(lines[:, 0, 2])
    y1 = max(lines[:, 0, 1])
    y2 = min(lines[:, 0, 3])

    # calculate line formula in y=k*x+n format
    k = (y1 - y2) / (x1 - x2)
    n = y1 - k * x1

    # calculate length of the line
    length = math.sqrt(pow((y2-y1), 2) + pow((x2-x1), 2))

    return Line(x1=x1, y1=y1, x2=x2, y2=y2, k=k, n=n, length=length)


def get_distance_from_line(line, point):
    """Gets orthogonal distance of point from line."""
    distance = abs((line.y2 - line.y1) * point.x - (line.x2 - line.x1) * point.y +
                   line.x2 * line.y1 - line.y2 * line.x1) / line.length
    return distance


def is_point_below_line(line, point):
    """Checks if point is below the line."""
    point_y = line.k * point.x + line.n
    return point_y > point.y
