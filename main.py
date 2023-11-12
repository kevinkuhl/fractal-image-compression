import argparse
import cv2
import numpy as np
from metrics import PSNR
from encode import compress
from decode import decompress
from plotting import plot_iterations
import time
import sys


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", type=str, help="The path to the image being compressed")
    parser.add_argument('img_type', choices=[
                        'grayscale', 'color'], help='The type of image being compressed')
    parser.add_argument("domain_bsize", type=int,
                        help="The size of the domain blocks")
    parser.add_argument("range_bsize", type=int,
                        help="The size of the range blocks")
    parser.add_argument("stride", type=int, help="The stride value in use")
    parser.add_argument("decomp_niter", type=int,
                        help="The number of iterations to use in decompression")
    parser.add_argument("rotate", type=str,
                        help="Option to apply block rotations")
    parser.add_argument("flip", type=str,
                        help="Option to apply block flipping")
    args = parser.parse_args()
    args.rotate = False if args.rotate.lower() == 'false' else True
    args.flip = False if args.flip.lower() == 'false' else True

    if args.img_type == 'grayscale':
        image = cv2.imread(args.path)
        image = np.mean(image[:, :, :2], 2)
        start_1 = time.time()
        fractal_code_book = compress(
            image, args.domain_bsize, args.range_bsize, args.stride, args.rotate, args.flip)
        end_1 = time.time()
        start_2 = time.time()
        iterations = decompress(fractal_code_book, args.domain_bsize,
                                args.range_bsize, args.stride, args.decomp_niter)
        end_2 = time.time()
        plot_iterations(iterations, image)
        print(f"Range Block size: {args.range_bsize}x{args.range_bsize}")
        print(f"Domain Block size: {args.domain_bsize}x{args.domain_bsize}")
        print(f"Number of iterations during decoding: {args.decomp_niter}")
        print(f"Time to encode:{end_1 - start_1} seconds")
        print(f"Time to decode: {end_2 - start_2} seconds")
        print(
            f"Size of the Fractal Code Book: {sys.getsizeof(fractal_code_book)}")
        print(f"Peak error: {np.max(iterations[-1]-image)}")
        value = PSNR(image, iterations[-1])
        print(f"PSNR value is {value} dB")

    elif args.img_type == 'color':
        raise NotImplementedError
