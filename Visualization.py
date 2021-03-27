from vars import CIFAR_Labels
import matplotlib.pyplot as plt


# Draw images
def visualization(x_rand, x_set, x_robust, y):
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
    plt.suptitle("Image class: " + CIFAR_Labels[y])
    plt.show()


# Plot errors
def error_plot():
    file_1 = open("cifar_nat_errors.txt", "r").readlines()
    file_2 = open("cifar_l2_0_5_errors.txt", "r").readlines()
    arr_1 = [float(i) for i in file_1]
    arr_2 = [float(i) for i in file_2]
    X = [i for i in range(0, 2001)]
    fig, ax = plt.subplots()
    ax.plot(X, arr_1, label="Epsilon = 0")
    ax.plot(X, arr_2, label="Epsilon = 0,5")
    ax.legend()
    plt.grid(True)
    plt.xlabel("Iterations")
    plt.ylabel("L2 norm of error")
    plt.show()


def test(x):
    fig = plt.figure(figsize=(6, 3))
    plt.subplots_adjust(wspace=0.3)
    first_image = fig.add_subplot(1, 1, 1)
    first_image.imshow(x)
    plt.show()
