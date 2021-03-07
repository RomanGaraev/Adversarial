from Attack import Attack
from Loader import CIFAR10, NumpyCIFAR10, ResNet50_l2_0_5_loader, ResNet50_l2_1_loader
from Model import ResNet50Feat, ResNet50
from Robust import create_robust

if __name__ == '__main__':
    # Test accuracy of pure model
    model = ResNet50(loader=ResNet50_l2_0_5_loader(CIFAR10()))
    dataset = NumpyCIFAR10()
    train, val = dataset.load()
    size = len(train[0])
    corr = 0
    for i in range(size):
        X = train[0][i: i+1].cuda()
        y = train[1][i]
        _, pred = model(X).topk(1, 1)
        corr += (pred == y)
    print(f"Accuracy: {(corr / size * 100)}")
    # Test PGD attack
    attack = Attack(model=model)
    attack.make_attack()

