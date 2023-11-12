## Python code for Fractal Image Compression

Usage:

    python3 main.py [-h] path {grayscale,color} domain_bsize range_bsize stride decomp_niter rotate flip

    Positional arguments:

          path               The path to the image being compressed
          
          {grayscale,color}  The type of image being compressed
          
          domain_bsize       The size of the domain blocks
          
          range_bsize        The size of the range blocks
          
          stride             The stride value in use
          
          decomp_niter       The number of iterations to use in decompression
          
          rotate             Option to apply block rotations
          
          flip               Option to apply block flipping
