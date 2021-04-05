from Vars import CIFAR_Labels

from numpy import arange, transpose, trace, array
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


def matrix_acc(conf):
    return f"Accuracy {trace(conf) / sum(sum(conf))}"


def confusion_mat(conf, labels=CIFAR_Labels):
    """
    Draw confusion matrix from
    :param conf: 2D-numpy matrix
    """
    fig, ax = plt.subplots()
    ax.imshow(conf)
    ax.set_xticks(arange(len(labels)))
    ax.set_yticks(arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    dims = conf.shape[0]
    # Loop over data dimensions and create text annotations.
    for i in range(dims):
        for j in range(dims):
            ax.text(j, i, conf[i, j], ha="center", va="center", color="w")
    fig.tight_layout()

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    plt.show()
    print(matrix_acc(conf))


if __name__ == "__main__":
    confusion_mat(array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), labels=["a", "b", "c"])