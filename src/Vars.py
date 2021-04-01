from os.path import join
from os import pardir
from torch import cuda


device = 'cuda' if cuda.is_available() else 'cpu'

# Amount of image samples, used in data loader
BATCH_SIZE = 4
# Threads for data loader
WORKERS = 4

# Used for visualisation
CIFAR_Labels = ["Airplane", "Automobile", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# Adversarial examples path
ATTACK_PATH = join(pardir, "datasets", "Attacks")
# Original CIFAR data set path
CIFAR_PATH = join(pardir, "datasets", "CIFAR10")
# Robust data set path, in numpy form
NUMPY_CIFAR_TRAIN = join(pardir, "datasets", "Train")
NUMPY_CIFAR_TEST = join(pardir, "datasets", "Test")
# Path of pretrained model
MODELS_PATH = join(pardir, "models")

# Steps of PGD
ROBUST_STEPS = 1000
# Amount of samples for shap training
SHAP_TRAIN_SIZE = 8
