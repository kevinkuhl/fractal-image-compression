import numpy as np

def PSNR(original, compressed):
    """
    Calculates the Peak Signal-to-Noise Ratio (PSNR) between an original and a compressed image.

    :param original: Original image as a NumPy array.
    :param compressed: Compressed image as a NumPy array.
    :return: PSNR value.
    """
    mse = np.mean((original - compressed) ** 2)
    if np.isclose(mse, 0):
        return float('inf')
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr