from robustness import model_utils, datasets
from vars import BATCH_SIZE, WORKERS, CIFAR_PATH, NUMPY_CIFAR_TRAIN, NUMPY_CIFAR_TEST
from torch.utils.data import Dataset, DataLoader, TensorDataset
from numpy import load, save, array
from torch import tensor
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Structure for robust set storing
class CustomSet(Dataset):
    def __init__(self):
        self.sample = []
        self.labels = []

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, item):
        return self.sample[item], self.labels[item]

    def add(self, X, y):
        for i in range(len(X)):
            self.sample.append(array(X[i]))
            self.labels.append(array(y[i]))

    def save(self, path=""):
        save(path + "X.npy", array(self.sample))
        save(path + "y.npy", array(self.labels))


# Base class for data set loading
class CustomLoader:
    def __init__(self):
        pass

    def load(self):
        pass

    def get_loaders(self):
        pass


''' Implementation of DataLoader '''


class CIFAR10(CustomLoader):
    def __init__(self):
        super(CIFAR10, self).__init__()

    def load(self):
        self.dataset = datasets.CIFAR(CIFAR_PATH)
        return self.dataset

    def get_loaders(self):
        self.load()
        train_loader, val_loader = self.dataset.make_loaders(batch_size=BATCH_SIZE, workers=WORKERS)
        return train_loader, val_loader


class NumpyCIFAR10(CustomLoader):
    def __init__(self):
        super(NumpyCIFAR10, self).__init__()

    def load(self):
        # [X_train, y_train], [X_test, y_test]
        return [tensor(load(NUMPY_CIFAR_TRAIN + "X.npy")), tensor(load(NUMPY_CIFAR_TRAIN + "y.npy"))], \
               [tensor(load(NUMPY_CIFAR_TEST + "X.npy")), tensor(load(NUMPY_CIFAR_TEST + "y.npy"))]

    def get_loaders(self):
        train, test = self.load()
        x_train, y_train = train[0], train[1]
        x_test, y_test = test[0], test[1]
        print("Numpy dataset is loaded.")
        return DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, num_workers=WORKERS), \
               DataLoader(dataset=TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, num_workers=WORKERS)


# TODO in future
def get_RestrictedImageNet():
    pass


# Loaders from robustness package
def get_loaders(dataset=CustomLoader()):
    train_loader, val_loader = dataset.make_loaders(batch_size=BATCH_SIZE, workers=WORKERS)
    return train_loader, val_loader


# Base class for loading different models from different sources
class ModelLoader:
    def __init__(self):
        pass

    def load(self):
        pass


'''Implementations of ModelLoader'''


# Epsilon = 0
class ResNet50_simple_loader(ModelLoader):
    def __init__(self, dataset=CustomLoader()):
        super(ResNet50_simple_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super(ResNet50_simple_loader, self).load()
        pretrained_model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=self.dataset.load(),
                                                                 resume_path='models\\cifar_nat.pt')
        return pretrained_model


# Epsilon = 0.5
class ResNet50_l2_0_5_loader(ModelLoader):
    def __init__(self, dataset=CustomLoader()):
        super(ResNet50_l2_0_5_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super().load()
        pretrained_model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=self.dataset.load(),
                                                                 resume_path='models\\cifar_l2_0_5.pt')
        return pretrained_model


# Epsilon = 1
class ResNet50_l2_1_loader(ModelLoader):
    def __init__(self, dataset=CustomLoader()):
        super(ResNet50_l2_1_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super().load()
        pretrained_model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=self.dataset.load(),
                                                                 resume_path='models\\cifar_l2_1_0.pt')
        return pretrained_model
