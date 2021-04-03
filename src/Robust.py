from Vars import ROBUST_STEPS, device
from Visualization import compare
import Loader
import Model
from torch import norm, renorm, clamp
from tqdm.auto import tqdm


def PGD_optim_l2(x, eps=0.5, alpha=0.1):
    """
    Project gradient descent optimization step

    :param x: noisy image that we want to modify
    :param eps: restriction of PGD perturbation
    :param alpha: step of gradient shifting
    :return: image in range [0, 1], shifted from x and projected to eps-ball by euclidean norm
    """
    # Step from https://github.com/MadryLab/robustness/blob/master/robustness/attack_steps.py
    # but with inverted direction
    l = len(x.shape) - 1
    g = x.grad.data
    g_norm = norm(g.view(g.shape[0], -1), dim=1).view(-1, *([1] * l))
    scaled_g = g / (g_norm + 1e-10)
    # Shift to initial image
    x_adv = x.detach() - alpha * scaled_g

    # l-2 projection to eps-ball
    proj = renorm(x_adv - x, dim=0, p=2, maxnorm=eps)

    # Valid image with values from 0 to 1
    clapped = clamp(x + proj, min=0, max=1).detach()
    clapped.requires_grad = True
    return clapped


def PGD_optim_linf(x, eps=0.5, alpha=0.1):
    """
    Project gradient descent optimization step

    :param x: noisy image that we want to modify
    :param eps: restriction of PGD perturbation
    :param alpha: step of gradient shifting
    :return: image in range [0, 1], shifted from x and projected to eps-ball by infinite norm
    """
    grad_sign = x.grad.data.sign()
    x_adv = x.detach() - alpha * grad_sign
    proj = clamp(x_adv - x, min=-eps, max=eps)
    clapped = clamp(x + proj, min=0, max=1).detach()
    clapped.requires_grad = True
    return clapped


def create_robust(model=Model.CustomModel, data_loader=Loader.CustomSetLoader, offset=0, plot=False):
    """
    Implementation of robust data set creation
    from "Adversarial Examples are not Bugs, they are Features"
    https://openreview.net/pdf/665eaf462c7aae47f0d7720f888add00dcbb24f3.pdf

    :param model: what model to use to extract robust features, i.e. ResNet50Feat
    :param data_loader: what data set should be modified, i.e. CIFAR10()
    :param offset: from which position of data set start the function
    :param plot: bool, draw or not initial, target and robust images
    :return: CustomSet, lists of robust images and their labels
    """

    robust_set = Loader.CustomSet()
    # [0] == train set loader
    loader = data_loader.get_loaders()['train']
    iterator = iter(loader)
    inner_i = 1
    # Current image and label, will be update in loop
    x_tar, y_tar = next(iterator)
    for i in range(offset):
        inner_i += 1
        x_tar, y_tar = next(iterator)
    # Future candidate image to robust_set: random image from data set,
    # because data set is shuffled and next image is random
    for x_new, y_new in iterator:
        device.empty_cache()
        model.zero_grad()
        # Copy random image to let it safe
        x_copy = x_new.clone()
        x_copy.requires_grad = True

        # Representation of original image
        g1 = model(x_tar.to(device))
        bar = tqdm(range(ROBUST_STEPS), desc=f"Iteration {inner_i} of {len(loader)}")
        inner_i += 1
        for _ in bar:
            bar.set_postfix({"Overall difference": format(norm(x_tar - x_copy).item(), '.5f')})
            # Representation of random image
            g2 = model(x_copy.to(device))
            # Features(orig) - Features(rand): L-2 loss between two penultimate layers g1 and g2
            loss = norm(g1 - g2, p=2)
            loss.backward(retain_graph=True)
            x_copy = PGD_optim_l2(x_copy)

        x_copy.requires_grad = False
        # Add modified image and original label
        robust_set.add(x_copy, y_tar)

        if plot:
            compare(x_new[0].clone(), x_tar[0].clone(), x_copy[0].clone(), y_tar[0].data)
        # Now let's modify this image
        x_tar, y_tar = x_new, y_new
        # Checkpoint, by default will be saved to NUMPY_CIFAR_TRAIN
        if inner_i % 10 == 0:
            robust_set.save()
    robust_set.save()
    return robust_set


def create_non_robust(model=Model.CustomModel, data_loader=Loader.CustomSetLoader):
    raise NotImplementedError


if __name__ == '__main__':
    model = Model.ResNet50Feat(loader=Loader.ResNet50_l2_0_5_loader(Loader.CIFAR10()))
    create_robust(model=model, data_loader=Loader.CIFAR10(), offset=100)
