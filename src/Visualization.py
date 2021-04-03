from Vars import CIFAR_Labels
from numpy import arange, transpose
import matplotlib.pyplot as plt


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


def confusion_mat(conf):
    """
    Draw confusion matrix from
    :param conf:
    :return:
    """
    fig, ax = plt.subplots()
    im = ax.imshow(conf)
    # We want to show all ticks...
    ax.set_xticks(arange(len(CIFAR_Labels)))
    ax.set_yticks(arange(len(CIFAR_Labels)))
    # ... and label them with the respective list entries
    ax.set_xticklabels(CIFAR_Labels)
    ax.set_yticklabels(CIFAR_Labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(CIFAR_Labels)):
        for j in range(len(CIFAR_Labels)):
            text = ax.text(j, i, conf[i, j],
                           ha="center", va="center", color="w")
    fig.tight_layout()
    plt.show()

