from Loader import CIFAR10, CustomSetLoader, CustomSet
from Model import CustomModel, ResNet18
from Vars import ATTACK_PATH, device
from art.estimators.classification import PyTorchClassifier
from art.attacks.evasion import PixelAttack
from torch import nn, optim
from numpy import argmax
from tqdm import tqdm


class Attack:
    def __init__(self, model=CustomModel, data_loader=CustomSetLoader().get_loaders()):
        """
        Class for creating adversarial examples by different attacks from foolbox
        :param model: target model, e.g. ResNet50()
        :param data_loader: target data set loader, e.g. CIFAR10().get_loaders()['train']
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        model.eval()
        self.f_model = PyTorchClassifier(model=model, input_shape=(3, 32, 32), clip_values=(0, 1),
                                         loss=criterion, optimizer=optimizer, nb_classes=10)
        self.loader = data_loader
        self.adv_examples = CustomSet()

    def make_attack(self):
        attack = PixelAttack(classifier=self.f_model)
        bar = tqdm(self.loader)
        print("Start creating adversarial examples...")

        Y = []
        Y_pred = []
        for X, y in bar:
            clipped = attack.generate(X.detach().cpu().numpy())
            pred = argmax(self.f_model.predict(clipped), axis=1)
            Y_pred.extend(pred)
            Y.extend(y)
            self.adv_examples.add(clipped, y.detach().cpu().numpy())

        print("Adversarial examples are created!")
        return self.adv_examples

    def save(self, path=ATTACK_PATH):
        self.adv_examples.save(path)


if __name__ == "__main__":
    at = Attack(model=ResNet18(), data_loader=CIFAR10().get_loaders()['train'])
    at.make_attack()