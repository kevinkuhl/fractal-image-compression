import matplotlib.pyplot as plt
import numpy as np
import math


def plot_iterations(iterations, target=None):
    """
    Plots each image in the 'iterations' list in a grid layout. If 'target' is provided,
    displays the RMSE between each image and the target.

    :param iterations: List of images (NumPy arrays) to be plotted.
    :param target: Optional target image to compute RMSE against each image in iterations.
    """
    plt.figure()
    nb_row = math.ceil(math.sqrt(len(iterations)))

    for i, img in enumerate(iterations):
        plt.subplot(nb_row, nb_row, i + 1)
        plt.imshow(img, cmap='gray', vmin=0, vmax=255, interpolation='none')
        title = str(i)
        if target is not None:
            title += ' (' + format_rmse(img, target) + ')'
        plt.title(title)
        hide_axes(plt.gca())

    plt.tight_layout()
    plt.savefig('iterations.png')


def format_rmse(image, target):
    """
    Calculates the RMSE between two images.

    :param image: Image as a NumPy array.
    :param target: Target image as a NumPy array.
    :return: Formatted RMSE as a string.
    """
    rmse = np.sqrt(np.mean(np.square(target - image)))
    return '{0:.2f}'.format(rmse)


def hide_axes(ax):
    """
    Hides the axes of a matplotlib subplot.

    :param ax: The axes of a matplotlib subplot.
    """
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
