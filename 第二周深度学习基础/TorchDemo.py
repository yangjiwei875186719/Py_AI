# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json
import matplotlib.pyplot as plt

"""

基于pytorch框架编写模型训练
实现一个自行构造的找规律(机器学习)任务
规律：x是一个5维向量，如果第1个数>第5个数，则为正样本，反之为负样本

"""


class TorchModel(nn.Module):  # 类 对象   nn.Module所有网络基类，管理网络属性（各种层，卷积层）
    def __init__(self, input_size):  # 定义方法
        super(TorchModel, self).__init__()
        # 初始化方法参数
        self.linear = nn.Linear(input_size, 1)  # 线性层  y = Ax + b，其中 A 是权重矩阵，x 是输入向量，b 是偏置向量，y 是输出向量。 默认两个，一个是输入向量，一个是输出向量
        self.activation = torch.sigmoid  # 激活函数：sigmoid归一化函数  # 用于将任何实数映射到介于0和1之间的值
        self.loss = nn.functional.mse_loss  # 损失函数：mse_loss均方差  loss函数采用均方差损失   # nn.functional：函数具体实现，如卷积、池化，激活函数等

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    # forward就是专门用来计算给定输入，得到神经元网络输出的方法  前向转播
    # 两层转换 第一层用线性层，第二层是
    def forward(self, x, y=None):
        x = self.linear(x)  # (batch_size, input_size) -> (batch_size, 1)
        y_pred = self.activation(x)  # (batch_size, 1) -> (batch_size, 1)
        if y is not None:
            return self.loss(y_pred, y)  # 预测值和真实值计算损失 自己去计算  (真实值-预测值)**2
        else:
            return y_pred  # 输出预测结果


# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个5维向量，如果第一个值大于第五个值，认为是正样本，反之为负样本
def build_sample():
    x = np.random.random(5)   # 随机一个5位向量
    print('###############')
    print(x)
    print('###############')
    if x[0] > x[4]:
        return x, 1
    else:
        return x, 0


# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(total_sample_num): # 传的是几行 5列的数据
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = build_sample() # 一个样本一个样本执行
        X.append(x)   # 1 * 5  因为默认 x = 5
        Y.append([y]) # 1 * 1
    print("X",X)
    print("Y",Y)
    # 当函数参数为整形时，表示生成矩阵的维度，此时参数可以为多个变量
    return torch.FloatTensor(X), torch.FloatTensor(Y) #  均方差允许是浮点数，所以是torch.FloatTensor(Y)  [y]额外增加维度 例：20 * 1
    # 举例子
    # x= torch.FloatTensor(1，2)
    # print(x)
    # 2x= torch.FloatTensor(1,2,3)
    # print(x)
    # tensor([[9.1477e-41，4.7684e-06]])
    # tensor([[[0.0000e+00，0.0000e+00，0.0000e+00]，[0.0000e+00，0.0000e+00，7.2868e-44]]])

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval() #  model.eval() # 作用等同于 self.train(False)简而言之，就是评估模式。而非训练模式。
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)  #
    print('----')
    # print(x)  # 张量 100 * 5
    # print(y)  # 张量 100 * 1
    print('----')
    print("本次预测集中共有%d个正样本，%d个负样本" % (sum(y), test_sample_num - sum(y)))
    correct, wrong = 0, 0   # 初始化正确个数和错误个数
    with torch.no_grad():   # 暗示不用计算梯度，在pytorch中，torch.no grad()是一个上下文管理器，用于指示pytorch不会计算梯度，当你想要在测试时使用模型进行预测，而不需要计算梯度时，可以使用torch.no_grad()。这样可以减少内存开销，提高代码运行效率。在使用该上下文管理器时，pytorch将不会记录任何操作用于计算梯度，从而避免不必要的计算。
        y_pred = model(x)  # 模型预测 model.forward(x)
        print("y_pred",y_pred)
        for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比

            if float(y_p) < 0.5 and int(y_t) == 0: # int函数 正数向下取正四舍，负数向上取证 五入 y_t 第1个数>第5个数，则为正样本 为1  ，负样本为0
                correct += 1  # 负样本判断正确
            elif float(y_p) >= 0.5 and int(y_t) == 1:
                correct += 1  # 正样本判断正确
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)  # 返回准确率


def main():
    # 配置参数
    epoch_num = 20  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 5000  # 每轮训练总共训练的样本总数
    input_size = 5  # 输入向量维度
    learning_rate = 0.001  # 学习率
    # 建立模型
    model = TorchModel(input_size)  #
    # 选择优化器  也有SGD
    # 学习率（lr）：控制每次参数更新的步长大小，默认值为0.001。学习率是一个关键参数，它决定了模型在训练过程中参数更新的速度。较大的学习率可能导致模型在最优解附近震荡，而较小的学习率则可能导致训练过程过于缓慢。
    # 动量参数（betas）：包括两个值，beta1和beta2，分别用于计算梯度的一阶矩（平均梯度）和二阶矩（梯度平方的平均值）。这两个参数的默认值分别为0.9和0.999。动量参数有助于加速收敛过程，通过引入一阶和二阶矩的信息，Adam算法能够在不同的问题和数据集上表现出良好的性能。
    # 用于维持数值稳定性的极小值（eps）：一个很小的值，用于防止除以零的情况，从而维持数值稳定性。默认值为1e-8。这个参数的存在是为了避免在计算过程中出现除以零的情况，特别是在梯度接近零时，通过添加一个极小的值来避免数值不稳定。
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate) # model.parameters()会返回一个迭代器，对这个迭代器遍历可以依次得到神经网络中的参数，也就是w1，b1，w2，b2  Adam更新权重，如果lr不调，默认 1e-3 要往最小的方向走
    log = []
    # 创建训练集，正常任务是读取训练集
    train_x, train_y = build_dataset(train_sample) # 创建训练集 ，x 返回 每5个数据为一组的100组数据 100 * 5,y返回一组数据通过 任务规律返回是正样本还是负样本
    # 训练过程
    for epoch in range(epoch_num): # 轮数 一轮 要运行5000样本，每批次运行20个，要运行250个批次 1轮次 = 250个批 * 一批20个样本 = 500个样本
        model.train()  # 切换训练模式
        watch_loss = []
        for batch_index in range(train_sample // batch_size):  #   5000/20 =250
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size] # 取位置[0,20] 下一次[20,40] 左开右闭区间
            # print('输出',x)
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]

            loss = model(x, y)  # 计算loss model.forward(x,y) 计算损失值
            loss.backward()  # 计算梯度   # 预测值和损失值相差多少
            optim.step()  # 更新权重 公式跑完 告诉应该怎么走
            optim.zero_grad()  # 梯度归零  算下一个batch
            watch_loss.append(loss.item())   # 存储损失值 # .item()方法是，取一个元素张量里面的具体元素值并返回该值，可以将一个零维张量转换成int型或者float型，在计算loss，accuracy时常用到
        # print(loss.item())
        # 作用：
        # 1.item（）取出张量具体位置的元素元素值
        # 2.并且返回的是该位置元素值的高精度值
        # 3.保持原元素类型不变；必须指定位置
        # 4.节省内存（不会计入计算图）
        #import torch
        # loss =torch.randn(2,2)
        # print(loss)
        # print(loss[1,1])
        # print(loss[1,1].item())

        # tensor([[-2.0274, -1.5974],
        #         [-1.4775,  1.9320]])
        # tensor(1.9320)
        # 1.9319512844085693
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)  # 测试本轮模型结果   # 准确率
        # print("watch_loss开始")
        # print(watch_loss)
        # print("watch_loss结束")
        log.append([acc, float(np.mean(watch_loss))])  # [[0.7, 0.24888162744045259], [0.78, 0.23306162428855895], [0.85, 0.21853427255153657],...
    # 保存模型
    torch.save(model.state_dict(), "model.pt")  # 后缀不影响，二进制存储
    # 画图
    print(log)  # len(log) = 20
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线 准确率
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return


# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    model = TorchModel(input_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())  # 存储到字典

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%d, 概率值：%f" % (vec, round(float(res)), res))  # 打印结果


if __name__ == "__main__":
    main()
    # test_vec = [[0.07889086,0.15229675,0.31082123,0.03504317,0.18920843],
    #             [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],
    #             [0.78797868,0.67482528,0.13625847,0.34675372,0.19871392],
    #             [0.79349776,0.59416669,0.92579291,0.41567412,0.1358894]]
    # predict("model.pt", test_vec)
