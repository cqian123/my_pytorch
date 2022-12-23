import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms
import cv2
import numpy as np

# 定义超参数
BATCH_SIZE = 64 # 定义每批处理的数据
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") #定义用cpu还是GPU训练
EPOCHS = 10 #训练数据集的轮次

# 构建transfor(pipeline)，对图像进行处理
pipeline = transforms.Compose([
    transforms.ToTensor(),#将图片转换为tensor
    transforms.Normalize((0.1307,),(0.3081,)) #归一化，在模型出现过拟合时，降低模型的复杂度
])

# 加载数据
from torch.utils.data import DataLoader

# 下载数据集
train_set = datasets.MNIST("data",train = True, download = True, transform=pipeline)
test_set = datasets.MNIST("data",train = False, download = True, transform=pipeline)
#加载数据集
train_loader = DataLoader(train_set, batch_size = BATCH_SIZE, shuffle=True)#最后一位参数用于打乱图片
test_loader = DataLoader(test_set, batch_size = BATCH_SIZE,shuffle = True)

#插入代码，显示MNIST中的图片
#with open(",/data/MNIST/raw/train-images-idx-ubyte","rb") as f:
#    file =f.read()
#imagel = [int(str(item).encode('ascii'),16) for item in file[16:16+784]]
#print(imagel)
#imagel_np = np.array(imagel, dype = np.uint8).reshape(28,28, 1)
# print(imagel_np.shape)
#cv2.imvrite("digit.jpg",imagel_np)

# 构建网络模型
class Digit(nn.Module):
    # 定义构造方法
    def __init__(self):
        super().__init__()
        #卷积层，两层
        self.conv1 = nn.Conv2d(1,10,5) # 1：灰度图片的通道， 10：输出通道 5卷积核
        self.conv2 = nn.Conv2d(10,20,3) # 10输入通道 20输出通道 3卷积核
        #全连接层
        self.fc1 = nn.Linear(20*10*10,500) #20*10*10 输入通道，500:输出通道
        self.fc2 = nn.Linear(500,10) # 500：输入通道，10：输出通道
    # 定义前向传播方法
    def forward(self, x):
        input_size = x.size(0) # batch_size * 1(灰度) * 28 * 28
        # 调用第一个卷积层
        x = self.conv1(x) # 输入:batch*1*28*28, 输出:Batch*10*24*24  (28-5+1=24)
        # 加上非线性的激活函数，使函数特征更明显
        x = F.relu(x)
        # 池化层，对图片进行压缩,
        x = F.max_pool2d(x,2,2) # 输入：Batch*10*24*24,输出Batch*10*12*12
        # 调用第二个卷积层
        x = self.conv2(x) # 输入：Batch*10*12*12， 输出Batch*20*10*10  (12-3+1=10)
        # 激活函数
        x = F.relu(x)
        # 把函数flatten
        x = x.view(input_size, -1) # 拉平，把多维数据拉成1维， -1自动计算维度，20*10*10 = 2000
        # 送入全连接层
        x = self.fc1(x) # 输入: batch*2000 输出:batch*500
        x = F.relu(x)
        x =self.fc2(x) # 输入: batch*500 输出：batch*10
        #通过损失函数计算分类后的概率
        output = F.log_softmax(x,dim=1)

        return output

# 定义优化器
model = Digit().to(DEVICE)

optimizer = optim.Adam(model.parameters())

# 定义训练方法
def train_model(model, device, train_loader, optimizer, epoch):
    # 模型训练
    model.train()
    for batch_index, (data, target) in enumerate(train_loader):
        # 部署到DEVICE上去
        data, target = data.to(device),target.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        output = model(data)
        # 计算损失,使用交叉熵损失，适用于多分类任务，二分类用sigmod函数
        loss = F.cross_entropy(output, target)# 可以计算出预测值和实际值之间的差距
        # 找到概率值最大的下标,这里没用到，可以后面写
        # pred = output.max(1, keepdim = True) #pred = output.argmax(dim = 1)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()
        if batch_index % 3000 == 0 :
            print("Train Eopch: {} \t Loss:{:.6f}".format(epoch,loss.item()))


#定义测试方法
def test_model(model, device, test_loader):
    # 模型验证
    model.eval()
    # 统计正确率
    correct = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad(): # 不计算梯度，也不会进行反向传播
        for data, target in test_loader:
            # 部署到device
            data, target = data.to(device), target.to(device)
            # 测试数据
            output = model(data)
            # 计算测试损失
            test_loss += F.cross_entropy(output, target).item()
            # 找到概率最大值的下标
            pred = output.max(1, keepdim = True)[1] # 0对应值， 1对应索引
            # pred = torch.max(output,dim = 1)
            # pred = output.argmax(dim = 1)
            #累计正确率
            correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        print("Test --- Average loss : {:.4f} ,Accuracy : {:.3f}\n".format(test_loss,100.0*correct/len(test_loader.dataset)))

for epoch in range(1,EPOCHS + 1):
    train_model(model, DEVICE, train_loader, optimizer, epoch)
    test_model(model, DEVICE, test_loader)




