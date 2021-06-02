from Vars import MODELS_PATH, device
from torch_test import test
import Loader

from torchvision.models.resnet import resnet50
from os.path import join
from tqdm import tqdm
import torch


def train_step(train_loader, model, optimizer):
    torch.cuda.empty_cache()
    model.train()
    model.to(device)
    bar = tqdm(train_loader)
    criteria = torch.nn.CrossEntropyLoss()
    for X, y in bar:
        X, y = X.cuda(), y.cuda()
        out = model(X)
        loss = criteria(out, y)
        loss.backward()
        optimizer.step()
        bar.set_postfix({"Loss": format(loss, '.4f')})
        optimizer.zero_grad()
    return model


def adjust_learning_rate(optimizer, epoch, lr):
    if epoch >= 30:
        lr /= 10
    if epoch >= 45:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print("Learning rate is ", param_group['lr'])


def train(model, data_loader: Loader.CustomSetLoader):
    dict_load = data_loader.get_loaders()
    train_loader = dict_load['train']
    val_loader = dict_load['test']
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0005)
    for epoch in range(70):
        print("Epoch ", epoch)
        adjust_learning_rate(optimizer, epoch, learning_rate)
        model = train_step(train_loader, model, optimizer)
        test(model, val_loader)
        torch.save(model.state_dict(), join(MODELS_PATH, "resnet50.pt"))


if __name__ == "__main__":
    model50 = resnet50()
    model50.fc = torch.nn.Linear(2048, 10)
    train(model=model50, data_loader=Loader.NumpyCIFAR10())
