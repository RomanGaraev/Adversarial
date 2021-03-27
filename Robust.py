from Visualization import visualization
from vars import ROBUST_STEPS
import Loader
import Model
from torch import norm, renorm, clamp, cuda
from tqdm.auto import tqdm


# args:
#     x - noisy image that we want to modify
#   eps - restriction of PGD perturbation
# alpha - step of gradient shifting
def PGD_optim_l2(x, eps=0.5, alpha=0.1):
    # Step (from https://github.com/MadryLab/robustness/blob/master/robustness/attack_steps.py)
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


# Inf norm of projection - result is not the same as for l-2
def PGD_optim_linf(x, eps=0.5, alpha=0.1):
    grad_sign = x.grad.data.sign()
    x_adv = x.detach() - alpha * grad_sign
    proj = clamp(x_adv - x, min=-eps, max=eps)
    clapped = clamp(x + proj, min=0, max=1).detach()
    clapped.requires_grad = True
    return clapped


'''
Implementation of robust data set creation
from the paper "Adversarial Examples are not Bugs, they are Features"
https://openreview.net/pdf/665eaf462c7aae47f0d7720f888add00dcbb24f3.pdf
'''


# args:
#       model - what model to use to extract robust features, i.e. ResNet50Feat
# data_loader - what data set should be modified, i.e. CIFAR10()
#      offset - from which position of data set start the function
def create_robust(model=Model.CustomModel, data_loader=Loader.CustomLoader, offset=0):
    robust_set = Loader.CustomSet()
    # [0] == train set loader
    loader = data_loader.get_loaders()[0]
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
        cuda.empty_cache()
        model.zero_grad()
        # Copy random image to let it safe
        x_copy = x_new.clone()
        x_copy.requires_grad = True

        # Representation of original image
        g1 = model(x_tar.cuda())
        bar = tqdm(range(ROBUST_STEPS), desc=f"Iteration {inner_i} of {len(loader)}")
        inner_i += 1
        for _ in bar:
            bar.set_postfix({"Overall difference": format(norm(x_tar - x_copy).item(), '.5f')})
            # Representation of random image
            g2 = model(x_copy.cuda())
            # Features(orig) - Features(rand): L-2 loss between two penultimate layers g1 and g2
            loss = norm(g1 - g2, p=2)
            loss.backward(retain_graph=True)
            x_copy = PGD_optim_l2(x_copy)

        x_copy.requires_grad = False
        # Add modified image and original label
        robust_set.add(x_copy, y_tar)

        visualization(x_new[0].clone().permute(1, 2, 0).detach().numpy(),
                      x_tar[0].permute(1, 2, 0),
                      x_copy[0].permute(1, 2, 0).detach().numpy(),
                      y_tar[0].data)
        # Now let's modify this image
        x_tar, y_tar = x_new, y_new
        # By default will be save to NUMPY_CIFAR_TRAIN
        if inner_i % 10 == 0:
          robust_set.save()
    robust_set.save()


def create_non_robust(model=Model.CustomModel, data_loader=Loader.CustomSetLoader):
    raise NotImplementedError


if __name__ == '__main__':
    model = Model.ResNet50Feat(loader=Loader.ResNet50_l2_0_5_loader(Loader.CIFAR10()))
    create_robust(model=model, data_loader=Loader.CIFAR10(), offset=100)
