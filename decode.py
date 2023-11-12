from transforms import transform_block, rescale
import numpy as np
from tqdm import tqdm


def decompress(fractal_code_book, domain_bsize, range_bsize, stride, nb_iter=8):
    """
    Reconstructs an image from a fractal code book using iterative transformations.

    :param fractal_code_book: Fractal code book containing transformation parameters.
    :param domain_bsize: Size of the domain blocks.
    :param range_bsize: Size of the range blocks.
    :param stride: Stride for the blocks.
    :param nb_iter: Number of iterations for the reconstruction process.
    :return: List of images from each iteration.
    """
    print("Decoding")
    factor = domain_bsize // range_bsize
    height = len(fractal_code_book) * range_bsize
    width = len(fractal_code_book[0]) * range_bsize
    num_rows, num_cols = len(fractal_code_book), len(fractal_code_book[0])

    iterations = [np.random.randint(0, 256, (height, width))]
    cur_img = np.zeros((height, width))

    for i_iter in tqdm(range(nb_iter)):
        for i in range(num_rows):
            for j in range(num_cols):
                cur_img = apply_transform(
                    cur_img, i, j, fractal_code_book, iterations, stride, domain_bsize, range_bsize, factor)
        iterations.append(cur_img.copy())
        cur_img.fill(0)

    return iterations


def apply_transform(cur_img, row, col, code_book, iterations, stride, domain_bsize, range_bsize, factor):
    """
    Applies the transformation specified in the fractal code book to a block of the image.

    :param cur_img: Current state of the image being reconstructed.
    :param row: Row index in the fractal code book.
    :param col: Column index in the fractal code book.
    :param code_book: Fractal code book containing transformation parameters.
    :param iterations: List of previous iterations of the image.
    :param stride: Stride for the blocks.
    :param domain_bsize: Size of the domain blocks.
    :param factor: Scaling factor between domain and range block sizes.
    :return: Updated image.
    """
    k, l, flip, angle, contrast, brightness = code_book[row][col]
    domain_block = rescale(
        iterations[-1][k*stride:k*stride+domain_bsize, l*stride:l*stride+domain_bsize], factor)
    range_block = transform_block(
        domain_block, flip, angle, contrast, brightness)
    cur_img[row*range_bsize:(row+1)*range_bsize, col *
            range_bsize:(col+1)*range_bsize] = range_block
    return cur_img
