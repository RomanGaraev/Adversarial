from Loader import CIFAR10, CustomSetLoader, CustomSet
from Model import CustomModel, ResNet18
from Vars import ATTACK_PATH, device
from foolbox import PyTorchModel
from foolbox import attacks
from tqdm import tqdm


class Attack:
    def __init__(self, model=CustomModel, data_loader=CustomSetLoader().get_loaders()):
        #self.f_model = PyTorchModel(model, bounds=(-5, 5))
        self.f_model = PyTorchModel(model, bounds=(-1, 1))
        self.loader = data_loader
        self.adv_examples = CustomSet()

    def make_attack(self, attack=attacks.L2PGD(steps=100), epsilons=0.25):
        correct = 0
        adv_correct = 0
        all = 0
        print("Start creating adversarial examples...")
        bar = tqdm(self.loader)
        for X, y in bar:
            X, y = X.to(device), y.to(device)
            raw, clipped, is_adv = attack(self.f_model, X, y, epsilons=epsilons)
            adv_correct += is_adv.detach().cpu().sum()
            self.adv_examples.add(clipped.cpu().detach().numpy(), y.cpu().detach().numpy())
            all += len(y)
            bar.set_postfix({"Adversarial ": float(adv_correct) / all})
        print("Adversarial examples are created!")
        print(f"The clean accuracy is {correct / len(self.loader) * 100.}%")
        print(f"The adversarial accuracy is {adv_correct / len(self.loader) * 100.}%")
        return self.adv_examples

    def save(self, path=ATTACK_PATH):
        self.adv_examples.save(path)


if __name__ == "__main__":
    at = Attack(model=ResNet18(), data_loader=CIFAR10().get_loaders()['train'])
    a = attacks.L2DeepFoolAttack()
    at.make_attack()
    at.save()