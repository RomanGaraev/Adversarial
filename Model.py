import Loader
from vars import SHAP_TRAIN_SIZE
from numpy import random
import torch
import shap


# Base class for NN models
class CustomModel(torch.nn.Module):
    def __init__(self, loader=Loader.ModelLoader):
        super(CustomModel, self).__init__()
        self.model = loader.load()
        self.model.eval()

    def get_model(self):
        return self.model

    def forward(self, x):
        return self.model(x)


'''Implementation of models'''


# loader=Loader.ResNet50_simple_loader(Loader.CIFAR10())
class ResNet50(CustomModel):
    def __init__(self, loader=Loader.ModelLoader):
        super(ResNet50, self).__init__(loader)
        pretrained_model = super().get_model()
        self.loader = loader
        self.model = torch.nn.Sequential(pretrained_model.normalizer,
                                         pretrained_model.model, torch.nn.Softmax())

    def forward(self, x):
        return self.model(x)


# ResNet without penultimate layer
class ResNet50Feat(CustomModel):
    def __init__(self, loader=Loader.ModelLoader):
        super(ResNet50Feat, self).__init__(loader)
        pretrained_model = super().get_model()
        self.loader = loader
        # Normalizer hits creating robust - increasing error
        self.model = torch.nn.Sequential(pretrained_model.normalizer,
                                         *list(pretrained_model.model.children())[:-1], torch.nn.Flatten())

    def forward(self, x):
        return self.model(x)


class ResNet50SHAP(CustomModel):
    def __init__(self, data_loader=Loader.CustomLoader, loader=Loader.ModelLoader):
        super(ResNet50SHAP, self).__init__(loader)
        pretrained_model = super().get_model()
        self.loader = loader
        self.model = torch.nn.Sequential(pretrained_model.normalizer,
                                         *list(pretrained_model.model.children())[:-1], torch.nn.Flatten())
        # Train shap explainer
        train, _ = data_loader.load()
        x_train = train[0]
        background = x_train[random.choice(x_train.shape[0], SHAP_TRAIN_SIZE, replace=False)].cuda()
        self.explainer = shap.DeepExplainer(self.model, background)

    def forward(self, x):
        # SHAP(g(x))
        return self.explainer.shap_values(x)
