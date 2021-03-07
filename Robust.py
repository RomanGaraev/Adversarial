from Visualization import visualization
from vars import BATCH_SIZE, ROBUST_STEPS
import Loader
import Model
import torch


# L-2 loss between two penultimate layers g1 and g2
def my_loss(g1, g2):
    return torch.norm(g1 - g2, p=2)


# PGD optimizer
class PGDOptim(torch.optim.Optimizer):
    def __init__(self, param, lr=0.01, alpha=0.99, eps=1e-8, weight_decay=5e-4, momentum=0.9, centered=False):
        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)
        super(PGDOptim, self).__init__(param, defaults)
        self.x = param[0]
    # TODO
    def step(self):
        grad_sign = self.x.grad.data.sign()
        self.x = self.x.detach()
        self.x -= 0.01 * grad_sign
        torch.clamp(self.x, 0, 1, out=self.x)


# args: model - some implementation of CustomModel;
#       data_loader - what data set to modify, from Loader
def create_robust(model, data_loader):
    robust_set = Loader.CustomSet()
    loader = data_loader.get_loaders()[0]
    #iterator = iter(loader)
    for X, y in loader:
        model.zero_grad()
        # Future candidate image to robust_set
        #try:
        #    x_new = next(iterator)[0]
        #    print(x_new)
        #except:
        x_new = torch.rand(size=(BATCH_SIZE, 3, 32, 32)) / 6 + 0.5
        x_new.requires_grad = True
        # For visualisation
        noise = torch.tensor(x_new[0], requires_grad=False).permute(1, 2, 0).detach().numpy()
        g1 = model(X.cuda())
        optim = torch.optim.RMSprop(lr=0.01, params=[x_new])
        #optim = PGDOptim(lr=0.1, param=[x_new])
        plato = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, verbose=True)
        for i in range(ROBUST_STEPS):
            print(f"Overall error:{torch.norm(X - x_new).item()};\
                    step: {i} \\ {ROBUST_STEPS};\
                    one example error: {torch.norm(X[0] - x_new[0]).item()}")
            g2 = model(x_new.cuda())
            # Features(image) - Features(noise)
            loss = my_loss(g1, g2)
            loss.backward(retain_graph=True)
            # Adopt noise image to decrease cost
            optim.step()
            plato.step(loss)
            x_new.requires_grad = False
            torch.clamp(x_new, 0, 1, out=x_new)
            x_new.requires_grad = True
        x_new.requires_grad = False
        robust_set.add(x_new, y)
        visualization(noise, X[0].permute(1, 2, 0), x_new[0].permute(1, 2, 0).detach().numpy(), y[0].data)
    robust_set.save()
