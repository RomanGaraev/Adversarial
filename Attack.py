from vars import BATCH_SIZE, ATTACK_PATH
from Loader import CIFAR10, CustomSet, get_loaders
from Model import CustomModel
from foolbox import PyTorchModel
from foolbox.attacks import PGD


class Attack:
    def __init__(self, model=CustomModel, dataset=CIFAR10()):
        self.f_model = PyTorchModel(model, bounds=(-5, 5))
        self.dataset = dataset.load()
        self.adv_examples = CustomSet()

    def make_attack(self, attack=PGD(steps=10), epsilons=0.03, path=ATTACK_PATH):
        train_loader, val_loader = get_loaders(self.dataset)
        correct = 0
        adv_correct = 0
        i = 0
        set_size = len(train_loader.dataset) / BATCH_SIZE
        print("Start creating adversarial examples...")
        for X, y in train_loader:
            X, y = X.cuda(), y.cuda()
            raw, clipped, is_adv = attack(self.f_model, X, y, epsilons=epsilons)
            adv_correct += is_adv.detach().cpu().sum()
            self.adv_examples.add(clipped.cpu().detach().numpy(), y.cpu().detach().numpy())
            i += 1
            if i % 10 == 0:
                print(i, " of ", set_size, "...")
        print("Adversarial examples are created.")
        self.adv_examples.save(path=path)
        print(f"The clean accuracy is {1. * correct / len(val_loader.dataset) * 100.}%")
        print(f"The adversarial accuracy is {1. * adv_correct / len(val_loader.dataset) * 100.}%")
