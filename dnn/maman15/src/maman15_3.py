import torch
from torch import nn

import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


BATCH_SIZE = 16


def show_images_per_class(dataloader, class_names, n_per_class=5):
    images_per_class = {cls: [] for cls in range(len(class_names))}

    for images, labels in dataloader:
        for img, lbl in zip(images, labels):
            if len(images_per_class[lbl.item()]) < n_per_class:
                images_per_class[lbl.item()].append(img)
        # Break early if all classes are filled
        if all(len(lst) == n_per_class for lst in images_per_class.values()):
            break

    # Plotting
    fig, axs = plt.subplots(len(class_names), n_per_class, figsize=(n_per_class * 2, len(class_names) * 2))
    for cls_idx, cls_name in enumerate(class_names):
        for i in range(n_per_class):
            img = images_per_class[cls_idx][i]
            img = img.permute(1, 2, 0)  # CHW -> HWC
            axs[cls_idx, i].imshow(img)
            axs[cls_idx, i].axis('off')
            if i == 0:
                axs[cls_idx, i].set_title(cls_name, loc='left')
    plt.tight_layout()
    plt.show()


def main():
    transform = transforms.ToTensor()

    cifar10_train_ds = datasets.CIFAR10("../data/", train=True, download=True, transform=transform)
    cifar10_test_ds = datasets.CIFAR10("../data/", train=False, download=True, transform=transform)

    cf10_train_dl = DataLoader(cifar10_train_ds, batch_size=BATCH_SIZE, shuffle=True)
    cf10_test_dl = DataLoader(cifar10_test_ds, batch_size=BATCH_SIZE, shuffle=True)

    resnet18 = models.resnet18(pretrained=True)
    #print(resnet18)

    # show_images_per_class(cf10_train_dl, cifar10_train_ds.classes, 5)

    for i, c in enumerate(resnet18.children()):
        print(c)



if __name__ == '__main__':
    main()
