from Vars import CIFAR_Labels, PLOTS_PATH
from Loader import NumpyCIFAR10

from numpy import arange, transpose, trace
import matplotlib.pyplot as plt
from os.path import join


def compare(x_rand, x_set, x_robust, y):
    """
    Draw three images: random initial image, target image from data set and robust image,
    which started from the initial and optimized to the target, i.e. has the closest representation to it.
    y correspond to target image class
    """
    # 3x32x32 -> 32x32x3
    x_rand = transpose(x_rand.detach().numpy(), axes=(1, 2, 0))
    x_set = transpose(x_set.detach().numpy(), axes=(1, 2, 0))
    x_robust = transpose(x_robust.detach().numpy(), axes=(1, 2, 0))

    fig = plt.figure(figsize=(6, 3))
    plt.subplots_adjust(wspace=0.3)
    first_image = fig.add_subplot(1, 3, 1)
    second_image = fig.add_subplot(1, 3, 2)
    third_image = fig.add_subplot(1, 3, 3)
    first_image.set_xticks([])
    first_image.set_yticks([])
    second_image.set_xticks([])
    second_image.set_yticks([])
    third_image.set_xticks([])
    third_image.set_yticks([])
    first_image.set_title("Noise start image")
    second_image.set_title("Target image")
    third_image.set_title("Robust image")

    first_image.imshow(x_rand)
    second_image.imshow(x_set)
    third_image.imshow(x_robust)
    plt.suptitle("Image class: " + CIFAR_Labels[y.data])
    plt.show()


def image_show(image, label):
    """
    Draw only one image
    """
    plt.imshow(transpose(image, axes=(1, 2, 0)))
    plt.suptitle("Image class: " + CIFAR_Labels[label.data])
    plt.show()


def plot_spector(image, label):
    im_x = image.real
    im_y = image.imag
    plt.plot(im_x, im_y)
    plt.suptitle("Image class: " + CIFAR_Labels[label.data])
    plt.show()


def confusion_mat(conf, labels=CIFAR_Labels, save=False, case="PGD-100-linf-0,025"):
    """
    Draw confusion matrix from
    :param conf: 2D-numpy confusion matrix
    :param labels: string list of classes
    :param case: string, name of file to save: name of method-steps-norm-epsilon
    :param save: save plot or not
    """
    fig, ax = plt.subplots()
    ax.imshow(conf)
    ax.set_xticks(arange(len(labels)))
    ax.set_yticks(arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    dims = conf.shape[0]
    # Loop over data dimensions and create text annotations.
    for i in range(dims):
        for j in range(dims):
            ax.text(j, i, conf[i, j], ha="center", va="center", color="w")
    plt.title(case)
    fig.tight_layout()
    fig.show()
    if save:
        fig.savefig(join(PLOTS_PATH, case))
    print(f"Accuracy {trace(conf) / sum(sum(conf))}")

"""
 confusion_mat(array(
 [[796,  178,  920,  428,  483,  160,  157,  148, 1452,  278],
 [262,  828,  225,  262,  130,  142,  353,  142, 1048, 1608],
 [429,   43,  766,  934,  898,  572,  794,  243,  273,   48],
 [159,   38,  541,  875,  696, 1322,  783,  335,  179,   72],
 [243,   29,  940,  878,  561,  554,  920,  670,  148,   57],
 [96,   32,  715, 1775,  480,  845,  423,  497,   81,   56],
 [70,  115,  950, 1390, 1161,  409,  520,   92,  230,   63],
 [230,   56,  437,  636, 1525,  957,  241,  726,   72,  120],
 [1587,  388,  493,  544,  339,  182,  310,   88,  629,  440],
 [385, 1336,  221,  526,  219,  274,  295,  344,  884,  516]]), save=False, case="FGSM-1-linf-0,0314")"""


if __name__ == "__main__":
    data = NumpyCIFAR10()
    train = data.get_loaders()['train']
    for X, y in train:
        image_show(X[0], y[0])
        break
