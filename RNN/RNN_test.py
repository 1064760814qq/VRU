import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.autograd import Variable

data_csv = pd.read_csv('data1.csv', usecols=[1])

# 数据预处理
data_csv = data_csv.dropna()
dataset = data_csv.values
dataset = dataset.astype('float32')
max_value = np.max(dataset)
min_value = np.min(dataset)
scalar = max_value - min_value

data=dataset

dataset = list(map(lambda x: (x)/scalar, dataset))

#创建数据集，比如312，274 对应一个y值237， 下一组就是274 237 对应278
def create_dataset(dataset, look_back=2):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back)]
        dataX.append(a)
        dataY.append(dataset[i + look_back])
    return np.array(dataX), np.array(dataY)
# 创建好输入输出，输入是25组
data_X, data_Y = create_dataset(dataset)
# print(data_Y)
# 划分训练集和测试集，70% 作为训练集 本来想做测试的，发现不需要也可以，自己运行出来自己测试看
train_size = int(len(data_X)*0.7)
test_size = len(data_X) - train_size
train_X = data_X[:train_size]
train_Y = data_Y[:train_size]
test_X = data_X[train_size:]
test_Y = data_Y[train_size:]

train_X = train_X.reshape(-1, 1, 2)
train_Y = train_Y.reshape(-1, 1, 1)
# print(train_Y)
test_X = test_X.reshape(-1, 1, 2)

train_x = torch.from_numpy(train_X)
train_y = torch.from_numpy(train_Y)
# print(train_y)
test_x = torch.from_numpy(test_X)
# print(torch.randn(128,1))
# lstm = nn.LSTM(3, 3)
# inputs = [torch.randn(1, 3) for _ in range(5)]
# inputs = torch.cat(inputs).view(len(inputs), 1, -1)
# hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
# out, (h,c) = lstm(inputs, hidden)
# print('out2',out)
# print('h:',h)
# print('c:',c)
# 定义模型
#torch.nn.Module 是一个类 里面有48个函数，这里nn.Module是父类，这里的lstm_reg是子类
class lstm_reg(nn.Module):
    #首先要初始化
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(lstm_reg, self).__init__()
#这里是Rnn，会返回两个对象
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        #这里的Linear，全连接层的输入与输出都是二维张量，一般形状为[batch_size, size]，
        # 不同于卷积层要求输入输出是四维张量
        #如果是128*20， 用Linear(20,40)  结果就是128*40的矩阵 这里就是全连接层。相当于矩阵相乘一样
        #返回16*1的格式
        self.reg = nn.Linear(hidden_size, output_size)  # 回归
        # print(self.reg)
    def forward(self, x):
        # print(x)
        #这里主要是求x，因为rnn会返回两个对象，第二个对象是一个（h，c）类型，不需要这个东西 所以直接用_表示
        x, o = self.rnn(x)  # (seq, batch, hidden)
        # print('oooook')
        # print(o)
        # print(x)
        # print("OOOOOOOOOKKKKKKK")
        # print(o)
        #这里的x是一个三维的，
        # print((x.shape))
        #s=16,b=1,h=4，这里就是继承父类的Lstm
        # s, b, h = x.shape
        # print(s)
        # print(b)
        # print(x)
        #这里得出的 x 就是16*4的格式，二维的,16是由24*0.7来的
        # x = x.view(s * b, h)  # 转换成线性层的输入格式
        # print(x)
        # x=x.view(16,4)
        x = self.reg(x)
        # print('ok')
        # print(x)
        #这里的-1 是成列的排，而不是成行
        # x = x.view(s, b, -1)
        # print(x)
        return x

#这里的2,4 。其中的2是输入值的大小，是两个值预测一个。4是隐藏层大小。Linear全连接层 输入值就是4，最后返回一个值
net = lstm_reg(2, 4)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
# 开始训练，1500步
for e in range(1500):
    var_x = Variable(train_x)
    var_y = Variable(train_y)
    # 前向传播
    out = net(var_x)
    # print(out)
    #计算损失值，用真实值与前向传播算出的值进行公式计算
    loss = criterion(out, var_y)
    # 反向传播
    #将梯度初始化为零（因为一个batch的loss关于weight的导数是所有sample的loss关于weight的导数的累加和）
    optimizer.zero_grad()
    #反向传播，计算当前的梯度
    loss.backward()
    #根据梯度更新网络参数
    optimizer.step()
    if (e + 1) % 100 == 0:  # 每 100 次输出结果
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
net = net.eval()  # 转换成测试模式

data_X = data_X.reshape(-1, 1, 2)
data_X = torch.from_numpy(data_X)
 # print(data_X)
var_data = Variable(data_X)

pred_test = net(var_data)  # 测试集的预测结果
# # 改变输出的格式
pred_test = pred_test.view(-1).data.numpy()
pred_test=np.array(pred_test)
pred_test=np.insert(pred_test,0,1.2085903)
pred_test=np.insert(pred_test,1,1.113576)
# print(pred_test)
# 画出实际结果和预测的结果
print(pred_test)
print((type(pred_test)))

plt.plot(pred_test*226.4, 'r', label='prediction')
plt.plot(data, 'b', label='real')
plt.legend(loc='best')
plt.show()
