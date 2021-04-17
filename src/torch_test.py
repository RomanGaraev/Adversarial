from Visualization import confusion_mat
from Loader import NumpyCIFAR10, ResNet50_simple_loader, CIFAR10
from Model import ResNet18, ResNet50

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np


def test(model, dataloader):
    pred = []
    Y = []
    for X, y in tqdm(dataloader):
        y_pred = np.argmax(model(X).cpu().detach().numpy(), axis=1)
        pred.extend(y_pred)
        Y.extend(y.cpu().detach().numpy())
    acc = len([i for i, x in enumerate(pred) if Y[i] == x]) / len(pred)
    print(f"Accuracy: {acc}")
    return pred, Y


if __name__ == "__main__":
    pred, Y = test(model=ResNet50(loader=ResNet50_simple_loader()), dataloader=NumpyCIFAR10().get_loaders()['train'])
    confusion_mat(confusion_matrix(y_true=Y, y_pred=pred),save=True, case="Robust CIFAR-10 train set")
