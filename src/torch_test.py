from Visualization import confusion_mat
from Loader import ResNet50_simple_loader, CIFAR10
from Model import ResNet
from Vars import device

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
import numpy as np


def test(model, dataloader):
    model.to(device)
    predictions = []
    y_real = []
    for X, y in tqdm(dataloader):
        predictions.extend(np.argmax(model(X.cuda()).cpu().detach().numpy(), axis=1))
        y_real.extend(y.cpu().detach().numpy())
    acc = len([i for i, x in enumerate(predictions) if Y[i] == x]) / len(predictions)
    print(f"Accuracy: {acc}")
    return predictions, y_real


if __name__ == "__main__":
    pred, Y = test(model=ResNet(loader=ResNet50_simple_loader(dataset=CIFAR10())),
                   dataloader=CIFAR10().get_loaders()['test'])
    confusion_mat(confusion_matrix(y_true=Y, y_pred=pred), save=False, case="Regular CIFAR-10 test set")
