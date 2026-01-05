import numpy as np
from torchvision import datasets
from torchvision.transforms import ToTensor

def load_mnist_numpy(root="./data"):
    train = datasets.MNIST(root=root, train=True, download=True, transform=ToTensor())
    test  = datasets.MNIST(root=root, train=False, download=True, transform=ToTensor())

    X_train = train.data.numpy().astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)

    X_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)

    # for MLP: flatten
    X_train = X_train.reshape(len(X_train), -1)   # (60000, 784)
    X_test  = X_test.reshape(len(X_test), -1)

    return X_train, y_train, X_test, y_test

def load_mnist_images_numpy(root="./data"):
    train = datasets.MNIST(root=root, train=True, download=True, transform=ToTensor())
    test  = datasets.MNIST(root=root, train=False, download=True, transform=ToTensor())

    # images: (N, 28, 28) uint8 -> float32 in [0,1]
    X_train = train.data.numpy().astype(np.float32) / 255.0
    y_train = train.targets.numpy().astype(np.int64)

    X_test = test.data.numpy().astype(np.float32) / 255.0
    y_test = test.targets.numpy().astype(np.int64)

    # CNN expects NCHW: (N, C, H, W)
    X_train = X_train[:, None, :, :]  # add channel dim
    X_test  = X_test[:, None, :, :]

    return X_train, y_train, X_test, y_test