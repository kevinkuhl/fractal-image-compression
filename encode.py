import numpy as np
from tqdm import tqdm
from transforms import transform_blocks, optimize_lsq


def compress(img, domain_bsize, range_bsize, stride, rotate=False, flip=False):
    """
    Compress an image using fractal encoding.

    :param img: NumPy array of the image to be compressed.
    :param domain_bsize: Size of the domain blocks.
    :param range_bsize: Size of the range blocks.
    :param stride: Stride for the blocks.
    :return: A list representing the fractal code book.
    """
    print("Encoding")
    range_rows = img.shape[0] // range_bsize
    range_cols = img.shape[1] // range_bsize

    # Preallocate fractal_code_book
    fractal_code_book = [[None for _ in range(
        range_cols)] for _ in range(range_rows)]

    transformed_blocks = transform_blocks(
        img, domain_bsize, range_bsize, stride, rotate, flip)
    pbar = tqdm(total=range_rows * range_cols)

    for row in range(range_rows):
        for col in range(range_cols):
            fractal_code_book[row][col] = find_best_match(
                img, row, col, range_bsize, transformed_blocks)
            pbar.update(1)

    pbar.close()
    return fractal_code_book


def find_best_match(img, row, col, range_bsize, transformed_blocks):
    """
    Find the best matching block for a given range block.

    :param img: NumPy array of the image to be compressed.
    :param row: Row index of the range block.
    :param col: Column index of the range block.
    :param range_bsize: Size of the range block.
    :param transformed_blocks: List of transformed domain blocks.
    :return: Best matching block parameters.
    """
    min_residual = float('inf')
    best_match = None
    range_block = img[row * range_bsize:(row + 1) * range_bsize,
                      col * range_bsize:(col + 1) * range_bsize]

    for domain_row, domain_col, direction, angle, domain_block in transformed_blocks:
        alpha, beta = optimize_lsq(range_block, domain_block)
        domain_block = alpha * domain_block + beta
        residual = np.sum(np.square(range_block - domain_block))

        if residual < min_residual:
            min_residual = residual
            best_match = (domain_row, domain_col,
                          direction, angle, alpha, beta)

    return best_match
