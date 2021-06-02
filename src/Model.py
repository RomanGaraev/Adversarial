from Vars import SHAP_TRAIN_SIZE, device
import Loader

from shap import GradientExplainer
from torch import nn, stack
from torch.fft import fft
from numpy import random


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
class ResNet(CustomModel):
    def __init__(self, loader=Loader.ResNet50_0_loader()):
        super(ResNet, self).__init__(loader)
        pretrained_model = super().get_model()
        self.loader = loader
        self.model = nn.Sequential(pretrained_model, nn.Softmax())

    def forward(self, x):
        return self.model(x)


# ResNet without penultimate layer
class ResNetFeat(CustomModel):
    def __init__(self, loader=Loader.ResNet50_l2_0_5_loader()):
        super(ResNetFeat, self).__init__(loader)
        pretrained_model = super().get_model()

        squeezed = []
        for i in range(len(pretrained_model) - 1):
            squeezed.append(pretrained_model[i])

        squeezed.extend(list(pretrained_model[-1].children())[:-1])
        self.model = nn.Sequential(*squeezed, nn.Flatten())


    def forward(self, x):
        return self.model(x)


class ResNetSHAP(CustomModel):
    def __init__(self, data_loader: Loader.CustomSetLoader(), loader=Loader.ResNet50_l2_0_5_loader()):
        super(ResNetSHAP, self).__init__(loader)
        pretrained_model = super().get_model()
        squeezed = []
        for i in range(len(pretrained_model) - 1):
            squeezed.append(pretrained_model[i])

        squeezed.extend(list(pretrained_model[-1].children())[:-1])
        self.model = nn.Sequential(*squeezed, nn.Flatten())
        self.data = data_loader
        # Train shap explainer
        train, _ = self.data.load()
        choice = random.choice(len(train), SHAP_TRAIN_SIZE, replace=False)
        background = stack([train[i][0].to(device) for i in choice])
        self.explainer = GradientExplainer(model=self.model, data=background)

    def forward(self, x):
        # SHAP(g(x))
        return self.explainer.shap_values(x)


class ResNetFourier(CustomModel):
    def __init__(self, loader=Loader.ResNet50_l2_0_5_loader()):
        super(ResNetFourier, self).__init__(loader)
        pretrained_model = super().get_model()
        squeezed = []
        for i in range(len(pretrained_model) - 1):
            squeezed.append(pretrained_model[i])

        squeezed.extend(list(pretrained_model[-1].children())[:-1])
        self.model = nn.Sequential(*squeezed, nn.Flatten())
        self.filter = nn.Linear(in_features=32768, out_features=10)

    def forward(self, x):
        x = fft(self.model(x), norm="ortho")
        #x = self.filter(x.float())
        return x


if __name__ == "__main__":
    model = ResNet(loader=Loader.ResNet50_0_loader())
    print(model)
