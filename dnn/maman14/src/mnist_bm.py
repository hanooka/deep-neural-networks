import torch

from torchvision import transforms
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets


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
            nn.LayerNorm([32, h // 2, w // 2]),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LayerNorm([64, h // 4, w // 4]),
            nn.Dropout(0.5),

            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.LayerNorm([128, h // 4, w // 4]),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((h // 4) * (w // 4) * 128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.5),
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
            nn.LayerNorm([32, h // 2, w // 2]),
            nn.Dropout(0.5),

            nn.Conv2d(32, 64, 3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.LayerNorm([64, h // 4, w // 4]),
            nn.Dropout(0.5),

            nn.Conv2d(64, 128, 3, stride=1, padding='same'),
            nn.ReLU(),
            #nn.MaxPool2d(2, 2),
            nn.LayerNorm([128, h // 4, w // 4]),
            nn.Dropout(0.5),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear((h // 4) * (w // 4) * 128, 256),
            nn.ReLU(),
            nn.LayerNorm(256),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.classifier(x)
        return x



def main():
    transform = transforms.Compose([
        transforms.ToTensor(),  # Converts to float tensor and scales to [0, 1]
        transforms.Normalize((0.1307,), (0.3081,))  # Mean and std from MNIST stats
    ])

    train_data = datasets.MNIST('../MNIST_data', download=True, train=True, transform=transform)
    test_data = datasets.MNIST('../MNIST_data', download=True, train=False, transform=transform)
    print(train_data)
    print(test_data)
    print(train_data.data.shape)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1000)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()


    for epoch in range(1, 6):  # 5 epochs
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] - Loss: {loss.item():.4f}')


if __name__ == '__main__':
    main()