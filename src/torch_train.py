from Vars import MODELS_PATH
import Loader
import torch
from tqdm import tqdm
from os.path import join


def train(self, data_loader=Loader.CustomSetLoader):
    self.model.train()
    dict_load = data_loader.get_loaders()
    train_loader = dict_load['train']
    optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, patience=10, verbose=True)
    criteria = torch.nn.CrossEntropyLoss()
    final_loss = 0
    for epoch in range(70):
        print("Epoch ", epoch)
        bar = tqdm(train_loader)
        for X, y in bar:
            X, y = X.cuda(), y.cuda()
            out = self.model(X)
            loss = criteria(out, y)
            loss.backward()
            optimizer.step()
            bar.set_postfix({"Loss": format(loss, '.4f')})
            optimizer.zero_grad()
            final_loss = loss.item()
        scheduler.step(metrics=final_loss)
    self.model.eval()
    torch.save(self.model.state_dict(), join(MODELS_PATH, "resnet.pt"))