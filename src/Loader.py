from Vars import BATCH_SIZE, WORKERS, CIFAR_PATH, NUMPY_CIFAR_TRAIN, NUMPY_CIFAR_TEST, MODELS_PATH
from resnets import ResNet18

from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch import tensor, nn, load as ch_load
from torchvision import datasets, transforms, models
from numpy import load, save, array
from robustness import model_utils,datasets as rob_dataset
from os.path import join


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
        # Train, test sets
        return datasets.CIFAR10(root=CIFAR_PATH, transform=transforms.ToTensor(), train=True),\
               datasets.CIFAR10(root=CIFAR_PATH, transform=transforms.ToTensor(), train=False)

    def get_loaders(self):
        train_set, test_set = self.load()
        train_loader = DataLoader(train_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=WORKERS)
        test_loader = DataLoader(test_set, shuffle=False, batch_size=BATCH_SIZE, num_workers=WORKERS)
        return {"train": train_loader, "test": test_loader}


class NumpyCIFAR10(CustomSetLoader):
    def __init__(self):
        super(NumpyCIFAR10, self).__init__()

    def load(self):
        # [X_train, y_train], [X_test, y_test]
        return [tensor(load(join(NUMPY_CIFAR_TRAIN, "X.npy"))), tensor(load(join(NUMPY_CIFAR_TRAIN, "y.npy")))], \
               [tensor(load(join(NUMPY_CIFAR_TEST, "X.npy"))),  tensor(load(join(NUMPY_CIFAR_TEST, "y.npy")))]

    def get_loaders(self):
        train_set, test_set = self.load()
        x_train, y_train = train_set[0], train_set[1]
        x_test, y_test = test_set[0], test_set[1]
        print("Numpy dataset is loaded.")
        train = DataLoader(dataset=TensorDataset(x_train, y_train), batch_size=BATCH_SIZE, num_workers=WORKERS)
        test = DataLoader(dataset=TensorDataset(x_test, y_test), batch_size=BATCH_SIZE, num_workers=WORKERS)
        return {"train": train, "test": test}


# TODO in future
def get_RestrictedImageNet():
    pass


'''Implementations of ModelLoader'''


# Loader from simple pytorch
class ResNet50_simple_loader(ModelLoader):
    def __init__(self, dataset=CustomSetLoader):
        super(ResNet50_simple_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super(ResNet50_simple_loader, self).load()
        pretrained_model = models.resnet50()
        pretrained_model.fc = nn.Linear(2048, 10)
        pretrained_model.load_state_dict(ch_load(join(MODELS_PATH, "resnet50.pt")))
        return pretrained_model

# Epsilon = 0
class ResNet50_0_loader(ModelLoader):
    def __init__(self, dataset=rob_dataset.CIFAR(data_path=CIFAR_PATH)):
        super(ResNet50_0_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super(ResNet50_0_loader, self).load()
        pretrained_model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=self.dataset,
                                                                 resume_path=join(MODELS_PATH, "cifar_nat.pt"))
        return pretrained_model


# Epsilon = 0.5
class ResNet50_l2_0_5_loader(ModelLoader):
    def __init__(self, dataset=rob_dataset.CIFAR(data_path=CIFAR_PATH)):
        super(ResNet50_l2_0_5_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super(ResNet50_l2_0_5_loader, self).load()
        pretrained_model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=self.dataset,
                                                                 resume_path=join(MODELS_PATH, "cifar_l2_0_5.pt"))
        return pretrained_model


# Epsilon = 1
class ResNet50_l2_1_loader(ModelLoader):
    def __init__(self, dataset=rob_dataset.CIFAR(data_path=CIFAR_PATH)):
        super(ResNet50_l2_1_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super().load()
        pretrained_model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=self.dataset,
                                                                 resume_path=join(MODELS_PATH, "cifar_l2_1_0.pt"))
        return pretrained_model


class ResNet50_inf_loader(ModelLoader):
    def __init__(self, dataset=rob_dataset.CIFAR(data_path=CIFAR_PATH)):
        super(ResNet50_inf_loader, self).__init__()
        self.dataset = dataset

    def load(self):
        super(ResNet50_inf_loader, self).load()
        pretrained_model, _ = model_utils.make_and_restore_model(arch='resnet50', dataset=self.dataset,
                                                                 resume_path=join(MODELS_PATH, "cifar_linf_8.pt"))
        return pretrained_model


class ResNet18_loader(ModelLoader):
    def __init__(self):
        super(ResNet18_loader, self).__init__()

    def load(self):
        super(ResNet18_loader, self).load()
        checkpoint = ch_load(join(MODELS_PATH, "basic_training_with_robust_dataset"))
        pretrained_model = ResNet18()
        pretrained_model = nn.DataParallel(pretrained_model)
        pretrained_model.load_state_dict(checkpoint['net'])
        return pretrained_model


# Create test CIFAR data set in numpy form
if __name__ == "__main__":
    tr_load = CIFAR10().get_loaders()['test']
    test = CustomSet()
    for X, y in tr_load:
        test.add(X, y)
    test.save(path=MODELS_PATH)