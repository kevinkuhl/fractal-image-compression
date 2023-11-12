from scipy import ndimage
import numpy as np
from scipy import optimize


def rescale(img, factor):
    """
    Reduces the size of an image by a given factor.

    :param img: NumPy array of the image.
    :param factor: Factor by which the image size is to be reduced.
    :return: Reduced image as a NumPy array.
    """
    return np.mean(
        img.reshape(img.shape[0]//factor, factor,
                    img.shape[1]//factor, factor),
        axis=(1, 3)
    )


def rotate(img, angle):
    """
    Rotates an image by a given angle without changing its shape.

    :param img: NumPy array of the image to be rotated.
    :param angle: The angle of rotation in degrees.
    :return: Rotated image as a NumPy array.
    """

    return ndimage.rotate(img, angle, reshape=False)


def flip(img, direction):
    """
    Flips an image array in the specified direction.

    :param img: NumPy array of the image.
    :param direction: Direction to flip the image. Accepts 'vertical' or 'horizontal'.
    :return: Flipped image as a NumPy array.
    """
    if direction == 'vertical':
        return img[::-1, :]
    elif direction == 'horizontal':
        return img[:, ::-1]
    elif direction == None:
        return img
    else:
        raise ValueError("Invalid direction. Use 'vertical' or 'horizontal'.")


def transform_block(img, direction, angle, alpha=1.0, beta=0.0):
    """
    Applies a transformation to an image block by flipping, rotating, and then 
    applying a linear transformation.

    :param img: NumPy array of the image to be transformed.
    :param direction: Direction to flip the image ('vertical' or 'horizontal').
    :param angle: The angle of rotation in degrees.
    :param alpha: Scaling factor for the linear transformation.
    :param beta: Shift value for the linear transformation.
    :return: Transformed image as a NumPy array.
    """

    flipped_img = flip(img, direction)
    rotated_img = rotate(flipped_img, angle)
    return alpha * rotated_img + beta


def optimize_lsq(range_block, domain_block):
    """
    Solves a least squares problem to find the optimal linear transformation
    (scaling and shifting) to map the domain block to the range block.

    :param range_block: NumPy array of the range block.
    :param domain_block: NumPy array of the domain block.
    :return: Tuple of scaling and shifting factors.
    """
    A = np.column_stack((np.ones(domain_block.size),
                        domain_block.reshape(-1, 1)))
    b = range_block.reshape(-1)

    bounds = [(-np.inf, -1.0), (np.inf, 1.0)]

    x = optimize.lsq_linear(A, b, bounds=bounds).x

    return x[1], x[0]


def transform_blocks(img, domain_bsize, range_bsize, stride, rotate=False, flip=False):
    """
    Generates transformed blocks from an image based on the specified parameters.

    :param img: NumPy array of the image.
    :param domain_bsize: Size of the domain blocks.
    :param range_bsize: Size of the range blocks.
    :param stride: Stride for block processing.
    :param rotate: Boolean to enable/disable rotation.
    :param flip: Boolean to enable/disable flipping.
    :return: List of transformed blocks.
    """
    angles = [0, 90, 180, 270] if rotate else [0]
    directions = ['vertical', 'horizontal'] if flip else [None]

    candidates = [(direction, angle)
                  for direction in directions for angle in angles]
    factor = domain_bsize // range_bsize

    return [
        (k, l, direction, angle, transform_block(rescale(img[k*stride:k*stride+domain_bsize,
                                                             l*stride:l*stride+domain_bsize], factor), direction, angle))
        for k in range((img.shape[0] - domain_bsize) // stride + 1)
        for l in range((img.shape[1] - domain_bsize) // stride + 1)
        for direction, angle in candidates
    ]
