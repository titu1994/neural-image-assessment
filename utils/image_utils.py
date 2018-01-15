import numpy as np
from PIL import Image, ImageOps
import warnings

IMAGE_SIZE = 256
CROP_SIZE = 224


warnings.filterwarnings('ignore', r'Possibly corrupt EXIF data',
                        UserWarning, "PIL.TiffImagePlugin",0)
warnings.filterwarnings('ignore', r'Corrupt EXIF data',
                        UserWarning, "PIL.TiffImagePlugin",0)


def crop_center(img, height, width):
    h, w, c = img.shape
    dx = (h - height) // 2
    dy = (w - width) // 2

    y1 = dy
    y2 = y1 + height
    x1 = dx
    x2 = x1 + width
    img = img[y1:y2, x1:x2, :]

    return img


def resize_and_pad(filename, desired_size=IMAGE_SIZE):
    """ Resize this into an image of size (desired_size, desired_size),
        preserving aspect ratio and in the case of a non-square image,
        centering it and padding the smaller side.

    :param filename: image filename
    :param desired_size: width and height of the returned image in pixels
    :return: downscaled image
    """
    img = Image.open(filename)
    img.thumbnail((desired_size, desired_size), Image.LANCZOS)
    delta_w = desired_size - img.width
    delta_h = desired_size - img.height
    padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2),
               delta_h - (delta_h // 2))
    new_im = ImageOps.expand(img, padding)
    return new_im


def preprocess_for_evaluation(filename):
    """ resized and pads an image to 256x256, then center crops it to 224x224
        and normalizes the values to the range [-1..1]
    :param filename: image filename
    :return: image as numpy array
    """
    img = resize_and_pad(filename, desired_size=IMAGE_SIZE)
    x = crop_center(
        np.array(img, dtype=np.float32)[:, :, :3], CROP_SIZE, CROP_SIZE)
    x /= 127.5
    x -= 1.
    return x
