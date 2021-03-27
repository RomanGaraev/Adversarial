from vars import SHAP_TRAIN_SIZE, MODELS_PATH
import Loader
from os.path import join
from numpy import random
from tqdm import tqdm
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

    def train(self, data_loader):
        pass


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

    def train(self, data_loader=Loader.CustomSetLoader):
        self.model.train()
        train_loader, val_loader = data_loader.get_loaders()
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0005)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=3, eps=0.001, verbose=True)
        criteria = torch.nn.CrossEntropyLoss()

        for epoch in range(0):
            print("Epoch ", epoch)
            bar = tqdm(train_loader)
            for X, y in bar:
                X, y = X.cuda(), y.cuda()
                optimizer.zero_grad()
                out = self.model(X)
                loss = criteria(out, y)
                loss.backward()
                optimizer.step()
                scheduler.step(metrics=loss)
                bar.set_postfix({"Loss": format(loss, '.4f')})
        torch.save(self.model.state_dict(), join(MODELS_PATH, "resnet.pt"))
        # Evaluation
        self.validate(train_loader, desc="Train set")
        self.validate(val_loader, desc="Test set")

    def validate(self, loader, desc="Train"):
        corr = 0
        for X, y in tqdm(loader, desc=desc):
            _, pred = self.model(X.cuda()).topk(1, 1)
            corr += (pred == y.cuda())
        print(f"{desc} accuracy: {(corr / len(loader) * 100)}")


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
    def __init__(self, data_loader=Loader.CustomSetLoader, loader=Loader.ModelLoader):
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

