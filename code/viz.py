import numpy as np
from characters import signature_to_letter
import numbers


def get_bars(img_size=(50, 50), thickness=.05,
             ignore_diagonals=True, whole_word=True):

    bars = np.array([signature_to_letter(e_i, img_size, thickness)
            for e_i in np.eye(16)])

    if ignore_diagonals:
        bars = bars[:-4]

    if whole_word:
        bars = np.concatenate([bars] * 4)

    return bars


def draw_words(signatures, bars):

    num_words = len(signatures)
    num_bars = signatures.shape[1]
    assert bars.shape[0] == num_bars

    bar_img_size_x, bar_img_size_y = bars.shape[1:]

    bars_per_word = num_bars / 4

    letters = (signatures[:, :, np.newaxis, np.newaxis] *
            bars[np.newaxis, :, :]).reshape(num_words, 4, bars_per_word,
                                bar_img_size_x, bar_img_size_y).sum(2)

    # The letters are basically done after this operation,
    # but they leave ugly intersections, which we will attempt to remove

    non_zero_bars = (signatures > 0).astype(np.float64)
    normalizer = (non_zero_bars[:, :, np.newaxis, np.newaxis] *
                  bars[np.newaxis, :, :]).reshape(num_words, 4, bars_per_word,
                                    bar_img_size_x, bar_img_size_y).sum(2)

    corrective_mask = normalizer > 0.

    letters[corrective_mask] /= normalizer[corrective_mask]

    # we now pad the letters, so they do not touch each other

    

    
def pad(arr, padding=10):

    shp = np.array(arr.shape)
    if isinstance(numbers.Number, padding):
        padding = padding * np.ones_like(shp)
    else:
        padding = np.array(padding)

    new_shape = shp + 2 * padding

    new_array = np.zeros(new_shape)

    slices = [slice(p, -p or None) for p in padding]

    new_array[slices] = arr

    return new_array


def make_collage(images):
    """Takes a 2D array of images and arranges them in this order.
    A 2D array of images is a 4D array, indexing the images in the
    first two dimensions and the pixels in the last two dimensions"""

    image_shape = list(images.shape[2:])
    image_arrangement = list(images.shape[:2])
    poster = images.transpose(0, 2, 1, 3).\
        reshape(image_arrangement[0], image_shape[0], -1)
    poster = poster.reshape(image_arrangement[0] * image_shape[0],
                            image_arrangement[1] * image_shape[1])

    return poster

