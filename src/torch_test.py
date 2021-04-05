from Visualization import confusion_mat
from Loader import CIFAR10
from Model import ResNet18


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


def test_vis(model, dataloader):
    pred, Y = test(model, dataloader)
    confusion_mat(confusion_matrix(y_true=Y, y_pred=pred))


if __name__ == "__main__":
    print("Testing on regular CIFAR test set...")
    test_vis(model=ResNet18(), dataloader=CIFAR10().get_loaders()['train'])
    #print("Testing on robust CIFAR train set...")
    #test_vis(model=ResNet18(), dataloader=NumpyCIFAR10().get_loaders()['train'])
