from vars import BATCH_SIZE, WORKERS, CIFAR_PATH, NUMPY_CIFAR_TRAIN, NUMPY_CIFAR_TEST, MODELS_PATH
from torch.utils.data import Dataset, DataLoader, TensorDataset
from robustness import model_utils, datasets
from numpy import load, save, array
from torch import tensor
from os.path import join
from os import environ
environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Structure for robust set storing
class CustomSet(Dataset):
    def __init__(self):
        self.sample = []
        self.labels = []

    def __len__(self):
        return len(self.sample)

    def __getitem__(self, i):
        return self.sample[i], self.labels[i]

    def add(self, x, y):
        self.sample.extend(x)
        self.labels.extend(y)

    def save(self, path=NUMPY_CIFAR_TRAIN):
        save(join(path, "X.npy"), [array(i) for i in self.sample])
        save(join(path, "y.npy"), [array(i) for i in self.labels])


# Base class for data set loading
class CustomSetLoader:
    def __init__(self):
        pass

    def load(self):
        pass

    def get_loaders(self):
        pass


# Base class for loading different models from different sources
class ModelLoader:
    def __init__(self):
        pass

    def load(self):
        pass


''' Implementation of DataLoader '''


class CIFAR10(CustomSetLoader):
    def __init__(self):
        super(CIFAR10, self).__init__()

    def load(self):
        self.dataset = datasets.CIFAR(CIFAR_PATH)
        return self.dataset

    def get_loaders(self):
        self.load()
        train_loader, val_loader = self.dataset.make_loaders(batch_size=BATCH_SIZE, workers=WORKERS, shuffle_train=False)
        return train_loader, val_loader


class NumpyCIFAR10(CustomSetLoader):
    def __init__(self):
        super(NumpyCIFAR10, self).__init__()

    def load(self):
        # [X_train, y_train], [X_test, y_test]
        return [tensor(load(join(NUMPY_CIFAR_TRAIN, "X.npy"))), tensor(load(join(NUMPY_CIFAR_TRAIN, "y.npy")))], \
               [tensor(load(join(NUMPY_CIFAR_TEST, "X.npy"))),  tensor(load(join(NUMPY_CIFAR_TEST, "y.npy")))]

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


'''Implementations of ModelLoader'''


# Epsilon = 0
class ResNet50_simple_loader(ModelLoader):
    def __init__(self, dataset=CustomSetLoader):
        super(ResNet50_simple_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super(ResNet50_simple_loader, self).load()
        pretrained_model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=self.dataset.load(),
                                                                 resume_path=join(MODELS_PATH, "cifar_nat.pt"))
        return pretrained_model


# Epsilon = 0.5
class ResNet50_l2_0_5_loader(ModelLoader):
    def __init__(self, dataset=CustomSetLoader):
        super(ResNet50_l2_0_5_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super().load()
        pretrained_model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=self.dataset.load(),
                                                                 resume_path=join(MODELS_PATH, "cifar_l2_0_5.pt"))
        return pretrained_model


# Epsilon = 1
class ResNet50_l2_1_loader(ModelLoader):
    def __init__(self, dataset=CustomSetLoader):
        super(ResNet50_l2_1_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super().load()
        pretrained_model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=self.dataset.load(),
                                                                 resume_path=join(MODELS_PATH, "cifar_l2_1_0.pt"))
        return pretrained_model
