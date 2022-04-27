import numpy as np
import cv2
from scipy import signal
import project_constants

ID1 = '315488171'
ID2 = '314756297'

def bgr_image_to_rgb_image(bgr_image: np.ndarray) -> np.ndarray:
    """Convert Blue-Green-Red image to Red-Green-Blue image.

    Args:
        bgr_image: np.ndarray of shape: (height, width, 3).

    Returns:
        rgb_image: np.ndarray of shape: (height, width, 3). Take the input
        image and in the third dimension, swap the first and last slices.
    """
    rgb_image = bgr_image.copy()
    rgb_image = rgb_image[:, :, [2, 1, 0]]
    return rgb_image


def black_and_white_image_to_tiles(arr: np.ndarray, nrows: int,
                                   ncols: int) -> np.ndarray:
    """Convert the image to a series of non-overlapping nrowsXncols tiles.

    Args:
        arr: np.ndarray of shape (h, w).
        nrows: the number of rows in each tile.
        ncols: the number of columns in each tile.
    Returns:
        ((h//nrows) * (w //ncols) , nrows, ncols) np.ndarray.
    Hint: Use only shape, reshape and swapaxes to implement this method.
    """
    h, w = arr.shape
    return (arr.reshape(h // nrows, nrows, -1, ncols)
            .swapaxes(1, 2)
            .reshape(-1, nrows, ncols))


def image_tiles_to_black_and_white_image(arr: np.ndarray, h: int,
                                         w: int) -> np.ndarray:
    """Convert the series of tiles back to a hxw image.

    Args:
        arr: np.ndarray of shape (nTiles, nRows, nCols).
        h: the height of the original image.
        w: the width of the original image.
    Returns:
        (h, w) np.ndarray.
    Hint: Use only shape, reshape and swapaxes to implement this method.
    """
    n, nrows, ncols = arr.shape

    return (arr.reshape(h // nrows, -1, nrows, ncols)
            .swapaxes(1, 2)
            .reshape(h, w))


def create_grad_x_and_grad_y(
        input_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the gradients across the x and y-axes.

    Args:
        input_image: np.ndarray. Image array.
    Returns:
        tuple (Ix, Iy): The first is the gradient across the x-axis and the
        second is the gradient across the y-axis.

    Recipe:
    If the image is an RGB image, convert it to grayscale using OpenCV's
    cvtColor. Otherwise, the input image is already in grayscale.
    Then, create a one pixel shift (to the right) image and fill the first
    column with zeros.
    Ix will be the difference between the grayscale image and the shifted
    image.
    Iy will be obtained in a similar manner, this time you're requested to
    shift the image from top to bottom by 1 row. Fill the first row with zeros.
    Finally, in order to ignore edge pixels, remove the first column from Ix
    and the first row from Iy.
    Return (Ix, Iy).
    """
    # Get image dimensions
    if len(input_image.shape) == 2:
        # this is the case of a black and white image
        nof_color_channels = 1
        height, width = input_image.shape

    else:
        # this is the case of an RGB image
        nof_color_channels = 3
        height, width, _ = input_image.shape

    if nof_color_channels == 3:
        img_as_grayscale = cv2.cvtColor(input_image, cv2.COLOR_RGB2GRAY)
    elif nof_color_channels == 1:
        img_as_grayscale = input_image
    else:
        raise ValueError("Unknown input_image format")

    height, width = img_as_grayscale.shape

    image_shifted_x_axes = np.concatenate((np.zeros([height, 1]), img_as_grayscale[:, :-1]), axis=1)
    image_shifted_y_axes = np.concatenate((np.zeros([1, width]), img_as_grayscale[:-1, :]), axis=0)

    Ix = img_as_grayscale - image_shifted_x_axes
    Iy = img_as_grayscale - image_shifted_y_axes

    "Remove First Column Of Ix And First Row From Iy"

    Ix[:, 0] = 0
    Iy[0, :] = 0

    return Ix, Iy


def calculate_response_image(input_image: np.ndarray, K: float) -> np.ndarray:
    """Calculate the response image for input_image with the parameter K.

    Args:
        input_image: np.ndarray. Image array.
        K: float. the K from the equation: R ≈ det(M) −k∙[trace(M)] ^2
    Returns:
        response_image: np.ndarray of shape (h,w). The response image is R:
        R ≈ det(M) −k∙[trace(M)] ^2.

    Recipe:
    Compute the image gradient using the method create_grad_x_and_grad_y.
    The response image is given by:  R ≈ det(M) −k∙[trace(M)] ^2
    where:
        det(M) = Sxx ∙ Syy - Sxy^2
        trace(M) = Sxx + Syy
        Sxx = conv(Ix^2, g)
        Syy = conv(Iy^2, g)
        Sxy = conv(Ix ∙ Iy, g)
    Convolutions are easy to compute with signal.convolve2d method. The
    kernel g should be 5x5 ones matrix, and the mode for the convolution
    method should be 'same'.
    Hint: Use np.square, np.multiply when needed.
    """
    # compute Ix and Iy
    Ix, Iy = create_grad_x_and_grad_y(input_image)

    "Create Kernel Matrix"
    kernel_matrix = np.ones([5, 5])

    "Calc Ix square And square"

    Ix_square = np.square(Ix)
    Iy_square = np.square(Iy)

    "Calc Sxx Syy And Sxy By Conv With Ones Kernel"

    Sxx = signal.convolve2d(Ix_square, kernel_matrix, mode='same')
    Syy = signal.convolve2d(Iy_square, kernel_matrix, mode='same')
    Sxy = signal.convolve2d(np.multiply(Ix, Iy), kernel_matrix, mode='same')

    "Calc Response Matrix "

    det_m = np.multiply(Sxx, Syy) - np.square(Sxy)
    trace_m = Sxx + Syy

    response_image = det_m - np.multiply(K, np.square(trace_m))
    return response_image


def our_harris_corner_detector(input_image: np.ndarray, K: float,
                               threshold: float) -> np.ndarray:
    """Calculate the corners for input image with parameters K and threshold.
    Args:
        input_image: np.ndarray. Image array.
        K: float. the K from the equation: R ≈ det(M) −k∙[trace M] ^2
        threshold: float. minimal response value for a point to be detected
        as a corner.
    Returns:
        output_image: np.ndarray with the height and width of the input
        image. This should be a binary image with all zeros except from ones
        in pixels with corners.
    Recipe:
    (1) calculate the response image from the input image and the parameter K.
    (2) apply Non-Maximal Suppression per 25x25 tile:
     (2.1) convert the response image to  tiles of 25x25.
     (2.2) For each tile, create a new tile which is all zeros except from
     one value - the maximal value in that tile. Keep the maximal value in
     the same position it was in the original tile.
     Hint: use np.argmax to find the index of the largest response value in a
     tile and read about (and use): np.unravel_index.
    (3) Convert the result tiles-tensor back to an image. Use
    the method: image_tiles_to_black_and_white_image.
    (4) Create a zeros matrix of the shape of the original image, and place
    ones where the image from (3) is larger than the threshold.
    """
    response_image = calculate_response_image(input_image, K)

    "Get Image Shape"
    height, width = input_image.shape
    current_image_threshold = threshold

    "Split The Image To Tiles"
    response_image_tiles = black_and_white_image_to_tiles(response_image, project_constants.TILES_NUM_ROW,
                                                          project_constants.TILES_NUM_COL)

    "Create Empty Array For New Tiles"
    new_response_image_tiles = np.zeros(response_image_tiles.shape)

    "Loop Over Tiles"
    for idx, tile in enumerate(response_image_tiles):

        "Create New Zeros Tile"
        new_tile = np.zeros(tile.shape)

        "Find Max Value Index"
        max_value_idx = np.unravel_index(np.argmax(tile), tile.shape)

        "Find Max Value"
        max_value = tile[max_value_idx]

        "Save New Tile"
        new_tile[max_value_idx] = max_value
        new_response_image_tiles[idx, :, :] = new_tile

    "Create Image From Tiles"
    image_from_tiles = image_tiles_to_black_and_white_image(new_response_image_tiles, height, width)

    "Create Output Image - Filled With Zeros"
    output_image = np.zeros(response_image.shape)

    "Place Ones In Output Image By Boolean Indexing"
    logical_indexes_of_one = image_from_tiles >= current_image_threshold
    output_image[logical_indexes_of_one] = 1

    return output_image