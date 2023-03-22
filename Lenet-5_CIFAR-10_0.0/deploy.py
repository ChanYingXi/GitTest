import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import os
import torch
print(np.__version__)
print(torchvision.__version__)
import matplotlib.pyplot as plt
import time
import copy


data_dir = "./hymenoptera_data"
model_name = "resnet"
num_classes = 2
batch_size = 32
num_epochs = 15
feature_extract = True

input_size = 224

all_imgs = datasets.ImageFolder(os.path.join(data_dir, "train"), transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
]))
loader = torch.utils.data.DataLoader(all_imgs, batch_size, shuffle=True)


data_transforms = {
    "train": transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    "val": transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
image_datasets = {
    "train": datasets.ImageFolder(os.path.join(data_dir, "train"), data_transforms["train"]),
    "val": datasets.ImageFolder(os.path.join(data_dir, "val"), data_transforms["val"])
}
dataloaders_dict = {
    "train": torch.utils.data.DataLoader(image_datasets["train"], batch_size=batch_size, shuffle=True, num_workers=4),
    "val": torch.utils.data.DataLoader(image_datasets["val"], batch_size=batch_size, shuffle=True, num_workers=4)
}

unloader = transforms.ToPILImage()
plt.ion()


def imshow(tensor, title=None):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # 暂停一会等待plot更新


plt.figure()
imshow(img[4], title="Image")

