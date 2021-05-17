from Vars import SHAP_TRAIN_SIZE, device
import Loader

from shap import GradientExplainer
from numpy import random
from torch import nn, stack


# Base class for NN models
class CustomModel(nn.Module):
    def __init__(self, loader=Loader.ModelLoader):
        super(CustomModel, self).__init__()
        self.model = loader.load()
        self.model.eval()

    def get_model(self):
        return self.model

    def forward(self, x):
        return self.model(x)

    def __repr__(self):
        return str(self.model)


'''Implementation of models'''


# loader=Loader.ResNet50_simple_loader(Loader.CIFAR10())
class ResNet50(CustomModel):
    def __init__(self, loader=Loader.ResNet50_0_loader()):
        super(ResNet50, self).__init__(loader)
        pretrained_model = super().get_model()
        self.loader = loader
        self.model = nn.Sequential(pretrained_model, nn.Softmax())

    def forward(self, x):
        return self.model(x)


# ResNet without penultimate layer
class ResNet50Feat(CustomModel):
    def __init__(self, loader=Loader.ResNet50_l2_0_5_loader()):
        super(ResNet50Feat, self).__init__(loader)
        pretrained_model = super().get_model()
        self.loader = loader
        self.model = nn.Sequential(pretrained_model.normalizer,
                                   *list(pretrained_model.model.children())[:-1], nn.Flatten())

    def forward(self, x):
        return self.model(x)


class ResNet50SHAP(CustomModel):
    def __init__(self, data_loader=Loader.CustomSetLoader(), loader=Loader.ResNet50_l2_0_5_loader()):
        super(ResNet50SHAP, self).__init__(loader)
        pretrained_model = super().get_model()
        self.model = nn.Sequential(pretrained_model.normalizer,
                                   *list(pretrained_model.model.children())[:-1], nn.Flatten())
        self.data = data_loader
        # Train shap explainer
        train, _ = self.data.load()
        choice = random.choice(len(train), SHAP_TRAIN_SIZE, replace=False)
        background = stack([train[i][0].to(device) for i in choice])
        self.explainer = GradientExplainer(model=self.model, data=background)

    def forward(self, x):
        # SHAP(g(x))
        return self.explainer.shap_values(x)


class ResNet18(CustomModel):
    def __init__(self, loader=Loader.ResNet18_loader()):
        super(ResNet18, self).__init__(loader)
        pretrained_model = super().get_model()
        self.loader = loader
        self.model = nn.Sequential(pretrained_model, nn.Softmax())

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    model = ResNet50(loader=Loader.ResNet50_inf_loader())
    print(model)
