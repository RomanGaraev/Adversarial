from Loader import CustomSetLoader, CustomSet, CIFAR10, ResNet50_l2_0_5_loader, ResNet50_0_loader
from Model import CustomModel, ResNet, ResNetFeat
from Visualization import confusion_mat
from Vars import ATTACK_PATH

from art.estimators.classification import PyTorchClassifier
from sklearn.metrics import confusion_matrix
import art.attacks.evasion as ev
from numpy import argmax, inf
from torch import nn, optim
from tqdm import tqdm


class Attack:
    def __init__(self, model: CustomModel, data_loader=CustomSetLoader().get_loaders(),
                 inp_shape=(3, 32, 32), classes=10):
        """
        Class for creating adversarial examples by different attacks from foolbox
        :param model: target model, e.g. ResNet50()
        :param data_loader: target data set loader, e.g. CIFAR10().get_loaders()['train']
        :param inp_shape: shape of input image
        :param classes: output shape
        """
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.eval()
        self.f_model = PyTorchClassifier(model=model, input_shape=inp_shape, clip_values=(0, 1),
                                         loss=criterion, optimizer=optimizer, nb_classes=classes, device_type="gpu")
        self.loader = data_loader
        self.adv_examples = CustomSet()

    def make_attack(self, attack):
        bar = tqdm(self.loader)
        print("Start creating adversarial examples...")
        y = []
        y_pred = []
        for X, y in bar:
            clipped = attack.generate(X)
            pred = argmax(self.f_model.predict(clipped), axis=1)
            y_pred.extend(pred)
            y.extend(y)
            self.adv_examples.add(clipped, y.detach().cpu().numpy())

        print("Adversarial examples are created!")
        return confusion_matrix(y_true=y, y_pred=y_pred)

    def save(self, path=ATTACK_PATH):
        self.adv_examples.save(path)


def attacks_test(repeats=3, epsilon=0.25):
    at = Attack(model=ResNet(), data_loader=CIFAR10().get_loaders()['train'])
    model = at.f_model
    attacks = {"FGSM l2"     : ev.FastGradientMethod(estimator=model, norm=2, eps=epsilon),
               "FGSM linf"   : ev.FastGradientMethod(estimator=model, norm=inf, eps=epsilon),
               "PGD l2 100"  : ev.ProjectedGradientDescentPyTorch(estimator=model, norm=2, eps=epsilon, max_iter=100),
               "PGD l2 1000" : ev.ProjectedGradientDescentPyTorch(estimator=model, norm=2, eps=epsilon, max_iter=1000),
               "PGD linf"    : ev.ProjectedGradientDescentPyTorch(estimator=model, norm=inf, eps=epsilon, max_iter=100),
               "DeepFool"    : ev.DeepFool(classifier=model, epsilon=epsilon, batch_size=32, verbose=False),
               "C-W l2 100"  : ev.CarliniL2Method(classifier=model, max_iter=100,),
               "C-W linf 100": ev.CarliniLInfMethod(classifier=model, max_iter=100, eps=epsilon)
               }
    # "Pixel" : ev.PixelAttack(classifier=model)
    for _ in range(repeats):
        for name, attack in attacks.items():
            print(f"{name} attack in processing...")
            conf = at.make_attack(attack=attack)
            print(conf)
            #print(f"{matrix_acc(conf)} for {name}")


if __name__ == "__main__":
    #attacks_test()
    model = Attack(model=ResNet(loader=ResNet50_l2_0_5_loader(dataset=CIFAR10())),
                   data_loader=CIFAR10().get_loaders()['test'])
    model_f = model.f_model
    attack = ev.ProjectedGradientDescentPyTorch(estimator=model_f, norm=inf, eps=0.025, max_iter=5, verbose=False)
    test_model = ResNetFeat(loader=ResNet50_0_loader())
    confusion_mat(model.make_attack(attack=attack), case="PGD-linf-5", save=False)
    model.make_attack(attack=attack)
