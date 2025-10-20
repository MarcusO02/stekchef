import cv2
import numpy as np

def redMask(rgb_image_bgr, debug=False):
    """
    Takes a BGR image and returns a binary mask of red areas.
    """
    hsv = cv2.cvtColor(rgb_image_bgr, cv2.COLOR_BGR2HSV)

    #Tune too work with meat (beanbag)
    lower_red1 = np.array([0, 100, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 50])
    upper_red2 = np.array([179, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    if debug:
        cv2.imshow("Red Mask", mask)
        cv2.waitKey(1)

    return mask

def greenMask(rgb_image_bgr, debug=False):
    """
    Takes a BGR image and returns a binary mask of green areas.
    """
    hsv = cv2.cvtColor(rgb_image_bgr, cv2.COLOR_BGR2HSV)

    #Tune to work with gripper
    lower_green = np.array([40, 50, 50])
    upper_green = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, lower_green, upper_green)

    if debug:
        cv2.imshow("Green Mask", mask)
        cv2.waitKey(1)

    return mask

def blackMask(rgb_image_bgr, debug=False):
    """
    Takes a BGR image and returns a binary mask of black areas.
    """
    hsv = cv2.cvtColor(rgb_image_bgr, cv2.COLOR_BGR2HSV)

    #Tune to work with black areas
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 60])

    mask = cv2.inRange(hsv, lower_black, upper_black)

    if debug:
        cv2.imshow("Black Mask", mask)
        cv2.waitKey(1)

    return mask
