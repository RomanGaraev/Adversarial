from Visualization import confusion_mat
from Loader import NumpyCIFAR10, ResNet50_simple_loader, CIFAR10, ResNet50_l2_0_5_loader
from Model import ResNet18, ResNet50
from Vars import device

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np


def test(model, dataloader):
    model.to(device)
    pred = []
    Y = []
    for X, y in tqdm(dataloader):
        y_pred = np.argmax(model(X.cuda()).cpu().detach().numpy(), axis=1)
        pred.extend(y_pred)
        Y.extend(y.cpu().detach().numpy())
    acc = len([i for i, x in enumerate(pred) if Y[i] == x]) / len(pred)
    print(f"Accuracy: {acc}")
    return pred, Y


if __name__ == "__main__":
    pred, Y = test(model=ResNet50(loader=ResNet50_l2_0_5_loader(dataset=CIFAR10())),
                   dataloader=CIFAR10().get_loaders()['test'])
    confusion_mat(confusion_matrix(y_true=Y, y_pred=pred), save=False, case="Regular CIFAR-10 test set")
