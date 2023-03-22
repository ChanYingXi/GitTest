import torchvision as tv
import torchvision.transforms as transforms
from torchvision.transforms import ToPILImage
import torch as t

# 可以把Tensor转化为Image，方便可视化
show = ToPILImage()

# 先伪造一个图片的Tensor，用ToPILImage显示
fake_img = t.randn(3, 32, 32)

# 显示图片
show(fake_img)


cifar_dataset = tv.datasets.CIFAR10(
    root='./data/',
    train = True,
    download = False
)

imgdata, label = cifar_dataset[90]
print('label: '+ str(label))
print('imgdata的类型:' )



def dataloader(train):

    transformer = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5),
                             std=(0.5, 0.5, 0.5))
    ])

    cifar_dataset = tv.datasets.CIFAR10(
        root='./data/cifar-10-batches-py/',  #下载的数据集所在的位置
        train = train,  # 是否为训练集。
        download = False,  # 设置为True，不用再重新下载数据
        transform = transformer
    )

    loader = t.utils.data.DataLoader(
        cifar_dataset,
        batch_size=4,
        shuffle=True,  # 打乱顺序
        num_workers=2  # worker数为2
    )

    return loader


classes = ('plane' + 'car' + 'bird' + 'cat' + 'deer' + 'dog' + 'frog' + 'horse' + 'ship' + 'truck')

# 训练集和测试集的加载器
trainloader = dataloader(train=True)
testloader = dataloader(train=False)


dataiter = iter(trainloader)

# 返回四张照片及其label
images, labels = dataiter.next()

# 打印多张照片
show(tv.utils.make_grid(images))