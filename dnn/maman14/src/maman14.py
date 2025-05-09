import math
import torch

from torch import nn
from typing import Union
from collections import OrderedDict

from torcheval import metrics
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def init_weights_pre_relu(input_dim, output_dim):
    """ Since we're using RELU activation, we'll implement the `he` initialization.
    I have ignored bias initialization problems, as we've got no "real training".
    No consideration on imbalance etc.
    We can test this using statistics and run few simulations to approx results with expectancy
    """
    std = math.sqrt(2 / input_dim)
    weights = torch.randn((input_dim, output_dim)) * std
    return weights


class SplitLinear(nn.Module):
    def __init__(self, input_dim, verbose=False):
        super().__init__()
        self.verbose = verbose
        output_dim = input_dim
        assert input_dim % 2 == 0, f"input_dim: {input_dim} should be even."

        self.network = nn.Sequential(OrderedDict([
            ("l1", nn.Linear(input_dim // 2, output_dim // 2)),
            ("a1", nn.ReLU())
        ]))
        # Custom weights creation!
        he_weights = init_weights_pre_relu(input_dim // 2, output_dim // 2)
        he_weights.requires_grad = True
        custom_weight = nn.Parameter(he_weights)
        self.network.l1.weight = custom_weight

    def set_verbose(self, verbose):
        self.verbose = verbose

    def forward(self, x: torch.Tensor):
        assert x.shape[1] % 2 == 0, f"x.shape[1]: {x.shape[1]} should be even."
        x1, x2 = x.split(x.shape[1] // 2, dim=-1)
        if self.verbose:
            print(f"x1: {x1}\nx2: {x2}")
        out1, out2 = self.network(x1), self.network(x2)
        if self.verbose:
            print(f"out1: {out1}\nout2: {out2}")
        return torch.cat([out1, out2], dim=-1)


def q1():
    N = 2  # Batch size
    M = 4  # Features (1d)

    model = SplitLinear(M, verbose=True)
    x = torch.rand((N, M))

    print(x)
    y = model(x)
    print(y)
    print(x.shape)
    print(y.shape)
    print(f"Shapes equal: {x.shape == y.shape}")


class DropNorm(nn.Module):
    def __init__(self, input_dim: Union[tuple, list, int]):
        super().__init__()
        self.eps = 1e-16
        # We init params so that y_i = x_i, similarly to batch norm
        self.gamma = nn.Parameter(torch.ones(input_dim))
        self.beta = nn.Parameter(torch.zeros(input_dim))

    def dropout(self, x: torch.Tensor):
        # hard set of p to 0.5 like required.
        p = 0.5
        if not self.training:
            return x
        feature_shape = x.shape[1:]
        ele_num = math.prod(feature_shape)
        # bitwise check for `even` num
        assert ele_num & 1 == 0
        half_ele = ele_num // 2
        # Creating tensor with half 1 and half 0
        mask = torch.cat([torch.ones(half_ele, dtype=torch.float, device=x.device),
                          torch.zeros(half_ele, dtype=torch.float, device=x.device)])
        # Generate random permutation (to order the 1s and 0s) <=> shuffle
        perm = torch.randperm(ele_num, device=x.device)
        # Shuffle the mask, reshape to original feature shape
        mask = mask[perm].reshape(feature_shape)
        return x * mask / p, mask

    def normalize(self, x):
        # We want all dims EXCEPT the batch dim, to be included in the mean
        # meaning every sample will have its own mew, sig2, and eventually norm_x.
        dims = tuple(range(1, x.dim()))
        mew = torch.mean(x, dtype=torch.float32, dim=dims, keepdim=True)
        # std^2 | known also as `variance`
        sig2 = torch.sum((x - mew) ** 2, dim=dims, keepdim=True) / math.prod(x.shape[1:])
        norm_x = (x - mew) / torch.sqrt(sig2 + self.eps)
        return norm_x

    def forward(self, x):
        """ When training, we use dropout -> normalization and we mult with mask as requested
            (we must multiply again with the mask, as beta might not be 0, and we want 0s)
        When not training, we only use normalize(x)*gamma + beta."""
        if self.training:
            out1, mask = self.dropout(x)
            out2 = self.normalize(out1)
            # We multiply at mask again because parameters that were zeroed in dropout should stay zeroed
            out2 = (self.gamma * out2 + self.beta) * mask
        else:
            out2 = self.gamma * self.normalize(x) + self.beta
        return out2


class BasicNetwork(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """ 28 => 14 => 7 """
        super().__init__()
        c, h, w = input_shape
        self.input_shape = input_shape
        self.backbone = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            nn.LayerNorm([32, h // 2, w // 2]),

            nn.Conv2d(32, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.5),
            nn.LayerNorm([64, h // 4, w // 4]),

            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm([128, h // 4, w // 4]),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((h // 4) * (w // 4) * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.LayerNorm(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


class MySpecialNetwork(nn.Module):
    def __init__(self, input_shape=(1, 28, 28), num_classes=10):
        """ 28 => 14 => 7 """
        super().__init__()
        c, h, w = input_shape
        self.input_shape = input_shape
        self.backbone = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            DropNorm([32, h // 2, w // 2]),

            nn.Conv2d(32, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            DropNorm([64, h // 4, w // 4]),

            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            DropNorm([128, h // 4, w // 4]),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((h // 4) * (w // 4) * 128, 256),
            nn.ReLU(),
            DropNorm(256),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x


def norm_example():
    x = torch.arange(0, 3 * 10 * 10).reshape(3, 10, 10)
    print(x)
    # We want all dims EXCEPT the batch dim, to be included in the mean
    dims = tuple(range(1, x.dim()))
    mew = torch.mean(x, dtype=torch.float32, dim=dims, keepdim=True)
    sig2 = torch.sum((x - mew) ** 2, dim=dims, keepdim=True) / math.prod(x.shape[1:])
    eps = 1e-16

    norm_x = (x - mew) / torch.sqrt(sig2 + eps)
    print(norm_x)


def validation_loop(model, val_loader, loss_fn) -> (float, float):
    """ validation loop """
    val_loss = 0.
    metric = metrics.MulticlassAccuracy(device=DEVICE)
    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            loss = loss_fn(preds, y)
            metric.update(preds, y)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss, metric.compute()


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        valid_loader: DataLoader,
        loss_fn: nn.Module,
        epochs: int = 10,
        verbose: int = 1,
        verbose_batch: int = 1,
        lr: float = 1e-4,
        wd: float = 0.05) -> nn.Module:
    """
    Given train/validation set, train the model `epochs` epochs, and validates at each epoch over
    the validation set.
    Required metric is Accuracy.

    :param model:
    :param train_loader:
    :param valid_loader:
    :param task: Task (currently 'classification' or 'regression')
    :param epochs:
    :param verbose: [0, 1, 2] Level of printing information (0 None, 2 Max)
    :param verbose_batch: if verbose is 2, how many batches before printing metrices and loss.
    :param lr: learning rate
    :param wd: weight decay
    :return: a model
    """
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    metric = metrics.MulticlassAccuracy(device=DEVICE)
    for epoch in range(epochs):
        running_loss = 0.
        model.train()
        metric.reset()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            metric.update(preds, y)
            loss.backward()
            opt.step()
            running_loss += loss.item()

            # Print every `verbose_batch` batches
            if verbose >= 2 and i % verbose_batch == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], "
                      f"Step [{i}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}", sep=',')

        # End of epoch. Run validation and print outcomes
        avg_val_loss, metric_val = validation_loop(model, valid_loader, loss_fn)
        if verbose >= 1:
            print(f"Epoch [{epoch + 1:4}/{epochs}]", end=f", ")
            print(f"trn los: {running_loss / len(train_loader):8.4f},", f"trn acc: {metric.compute():6.4f}",
                  end=', ')
            print(f"val loss: {avg_val_loss:8.4f}, val acc: {metric_val:6.4f}")

    return model


def run_mnist(model: nn.Module):
    model = model.to(DEVICE)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # mean and std from MNIST stats
    ])
    train_data = datasets.MNIST('../MNIST_data', download=True, train=True, transform=transform)
    test_data = datasets.MNIST('../MNIST_data', download=True, train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=50, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=100)

    loss_fn = nn.CrossEntropyLoss()
    train_model(model, train_loader, test_loader, loss_fn, verbose=2, verbose_batch=100)


if __name__ == '__main__':
    run_mnist(BasicNetwork())
    run_mnist(MySpecialNetwork())
