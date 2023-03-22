# coding=utf-8
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torchvision import transforms, datasets, models
import shutil
from matplotlib import pyplot as plt
# 随机种子设置
random_state = 42
np.random.seed(random_state)
#接下来的数据是把原训练集90%的数据做训练，10%做测试集，其中把分为训练集的数据内的猫和狗分开，分为测试集的数据的猫和狗进行分开保存在新的各自的目录下
# kaggle原始数据集地址
original_dataset_dir = 'D:\\Code\\Python\\Kaggle-Dogs_vs_Cats_PyTorch-master\\data\\train'  #训练集地址
total_num = int(len(os.listdir(original_dataset_dir)) )  #训练集数据总数，包含猫和狗
random_idx = np.array(range(total_num))
np.random.shuffle(random_idx)#打乱图片顺序

# 待处理的数据集地址
base_dir = 'D:\\Code\\dogvscat\\data2' #把原训练集数据分类后的数据存储在该目录下
if not os.path.exists(base_dir):
    os.mkdir(base_dir)

# 训练集、测试集的划分
sub_dirs = ['train', 'test']
animals = ['cats', 'dogs']
train_idx = random_idx[:int(total_num * 0.9)] #打乱后的数据的90%是训练集，10是测试集
test_idx = random_idx[int(total_num * 0.9):int(total_num * 1)]
numbers = [train_idx, test_idx]
for idx, sub_dir in enumerate(sub_dirs):
    dir = os.path.join(base_dir, sub_dir)#'D:\\Code\\dogvscat\\data2\\train'或'D:\\Code\\dogvscat\\data2\\test'
    if not os.path.exists(dir):
        os.mkdir(dir)

    animal_dir = ""

    #fnames = ['.{}.jpg'.format(i) for i in numbers[idx]]
    fnames = ""
    if sub_dir == 'train':
        idx = 0
    else:
        idx =1
    for i in numbers[idx]:
        #print(i)
        if i>=12500:#把数据保存在dogs目录下
            fnames = str('dog'+'.{}.jpg'.format(i))
            animal_dir = os.path.join(dir,'dogs')

            if not os.path.exists(animal_dir):
                os.mkdir(animal_dir)
        if i<12500:#图片是猫，数据保存在cats目录下
            fnames = str('cat'+'.{}.jpg'.format(i))
            animal_dir = os.path.join(dir, 'cats')
            if not os.path.exists(animal_dir):
                os.mkdir(animal_dir)
        src = os.path.join(original_dataset_dir, str(fnames)) #原数据地址
        #print(src)
        dst = os.path.join(animal_dir, str(fnames))#新地址
        #print(dst)
        shutil.copyfile(src, dst)#复制


        # 验证训练集、测试集的划分的照片数目
    print(dir + ' total images : %d' % (len(os.listdir(dir+'\\dogs'))+len(os.listdir(dir+'\\cats'))))
    # coding=utf-8

# 配置参数
random_state = 1
torch.manual_seed(random_state)  # 设置随机数种子，确保结果可重复
torch.cuda.manual_seed(random_state)# #为GPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed_all(random_state) #为所有GPU设置种子用于生成随机数，以使得结果是确定的
np.random.seed(random_state)
# random.seed(random_state)

epochs = 10 # 训练次数
batch_size = 4  # 批处理大小
num_workers = 0  # 多线程的数目
use_gpu = torch.cuda.is_available()
PATH='D:\\Code\\dogvscat\\model.pt'
# 对加载的图像作归一化处理， 并裁剪为[224x224x3]大小的图像
data_transform = transforms.Compose([
    transforms.Resize(256),#重置图像分辨率
    transforms.CenterCrop(224), #中心裁剪
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  #归一化
])

train_dataset = datasets.ImageFolder(root='D:\\Code\\dogvscat\\data2\\train',
                                     transform=data_transform)
print(train_dataset)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           num_workers=num_workers)

test_dataset = datasets.ImageFolder(root='D:\\Code\\dogvscat\\data2\\test', transform=data_transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


# 创建模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.maxpool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 1024)
        torch.nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, 512)
        torch.nn.Dropout(0.5)
        self.fc3 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.maxpool(F.relu(self.conv1(x)))
        x = self.maxpool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


net = Net2()
if(os.path.exists('D:\\Code\\dogvscat\\model.pt')):
    net=torch.load('D:\\Code\\dogvscat\\model.pt')

if use_gpu:
    print('gpu is available')
    net = net.cuda()
else:
    print('gpu is unavailable')

print(net)
trainLoss = []
trainacc = []
testLoss = []
testacc = []
x = np.arange(1,11)
# 定义loss和optimizer
cirterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)

def train():

    for epoch in range(epochs):
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        for step, data in enumerate(train_loader, 0):#第二个参数表示指定索引从0开始
            inputs, train_labels = data
            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(train_labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(train_labels)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, train_predicted = torch.max(outputs.data, 1)  #返回每一行最大值的数值和索引，索引对应分类
            train_correct += (train_predicted == labels.data).sum()
            loss = cirterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_total += train_labels.size(0)

        print('train %d epoch loss: %.3f  acc: %.3f ' % (
            epoch + 1, running_loss / train_total, 100 * train_correct / train_total))
        # 模型测试
        correct = 0
        test_loss = 0.0
        test_total = 0
        test_total = 0
        net.eval() #测试的时候整个模型的参数不再变化
        for data in test_loader:
            images, labels = data
            if use_gpu:
                images, labels = Variable(images.cuda()), Variable(labels.cuda())
            else:
                images, labels = Variable(images), Variable(labels)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = cirterion(outputs, labels)
            test_loss += loss.item()
            test_total += labels.size(0)
            correct += (predicted == labels.data).sum()

        print('test  %d epoch loss: %.3f  acc: %.3f ' % (epoch + 1, test_loss / test_total, 100 * correct / test_total))
        trainLoss.append(running_loss / train_total)
        trainacc.append(100 * train_correct / train_total)
        testLoss.append(test_loss / test_total)
        testacc.append(100 * correct / test_total)
    plt.figure(1)
    plt.title('train')
    plt.plot(x,trainacc,'r')
    plt.plot(x,trainLoss,'b')
    plt.show()
    plt.figure(2)
    plt.title('test')
    plt.plot(x,testacc,'r')
    plt.plot(x,testLoss,'b')
    plt.show()



    torch.save(net, 'D:\\Code\\dogvscat\\model.pt')


train()