import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=20,
                               kernel_size=5,
                               stride=1)  # todo: whats the params?
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = nn.functional.relu(self.conv1(x))  # 1*28*28 -> 20*24*24
        x = nn.functional.max_pool2d(x, kernel_size=2, stride=2)  # -> 20*12*12
        x = nn.functional.relu(self.conv2(x))  # -> 50*8*8
        x = nn.functional.max_pool2d(x, 2, 2)  # -> 50*4*4
        x = x.view(-1, 4 * 4 * 50)  #
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return nn.functional.log_softmax(x, dim=1)

# 求平均值和标准差
data = [d[0].data.cpu().numpy() for d in mnist_data]
print(np.mean(data))
print(np.std(data))

batch_size = 32

train_dataloader = torch.utils.data.DataLoader(datasets.MNIST("./mnist_data", train=True, download=False,
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=(0.13066062,),std=(0.30810776,))
                                                    ])),
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=0,
                                               pin_memory=True)  # 把一些数据固定到内存中
test_dataloader = torch.utils.data.DataLoader(datasets.MNIST(
    "./mnist_data", train=False, download=False,
                                                    transform=transforms.Compose([
                                                        transforms.ToTensor(),
                                                        transforms.Normalize(mean=(0.13066062,),std=(0.30810776,))
                                                    ])),
                                               batch_size=batch_size,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True)
# Data loader. Combines a dataset and a sampler, and provides an iterable over
# the given dataset.

def train(model, train_loader, optimizer, epoch):
    model.train()
    for idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        pred = model(data)  # shape: batch_size * 10
        loss = nn.functional.nll_loss(pred, target)
        loss.backward()
        if idx % 10 == 0:
            print(f"Train Epoch: {epoch}, iteration: {idx}, Loss: {loss.item()}")


def test(model, test_loader):
    model.eval()
    total_loss = 0.
    correct = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)

            pred = model(data).argmax(dim=1)  # shape: batch_size * 1
            loss = nn.functional.nll_loss(pred, target, reduction='sum').item()
            correct += pred.eq(target.view_as(pred)).sum().item()

            if idx % 10 == 0:
                print(f"Test Epoch: {epoch}, iteration: {idx}, Loss: {loss.item()}")

    total_loss /= len(test_loader.dataset)
    acc = correct / len(test_loader.dataset) * 100
    print(f"Test loss: {total_loss}, Accuracy: {acc}")

lr = 0.01
momentum = 0.5
model = CNNModel().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

num_epochs = 2
for epoch in range(num_epochs):
    train(model, train_dataloader, optimizer, epoch)
    test(model, test_dataloader)

torch.save(model.state_dict(), "mnist_cnn.pytorch_model.bin")

