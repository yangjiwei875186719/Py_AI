# -*- coding: utf-8 -*-

import torch
import os
import random
import os
import numpy as np
import logging
from config import Config
from model import TorchModel, choose_optimizer
from evaluate import Evaluator
from loader import load_data
#[DEBUG, INFO, WARNING, ERROR, CRITICAL]
"""
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')：
logging.basicConfig() 函数用于配置默认的日志行为。
level=logging.INFO 指定了日志记录的级别为 INFO，这意味着只有 INFO 级别及以上的日志消息会被记录。
format='%(asctime)s - %(name)s - %(levelname)s - %(message)s' 指定了日志消息的格式，其中：
%(asctime)s 会被替换为日志记录的时间；
%(name)s 会被替换为记录器的名称；
%(levelname)s 会被替换为日志消息的级别（如 INFO、DEBUG 等）；
%(message)s 会被替换为实际的日志消息。
logger = logging.getLogger(__name__)：
logging.getLogger(__name__) 创建了一个名为 __name__ 的记录器对象，__name__ 是 Python 中的特殊变量，表示当前模块的名称。
这个记录器对象可以用来记录各种级别的日志消息，通过调用其方法如 logger.debug(), logger.info(), logger.warning(), logger.error(), logger.critical() 等来记录不同级别的日志信息。
https://www.yuque.com/linghunliulangzhe-prqjv/qdg60l/uq4e1qwew602clnp
"""
logging.basicConfig(level=logging.INFO, format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

"""
模型训练主程序
"""


seed = Config["seed"] 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

def main(config):
    #创建保存模型的目录
    if not os.path.isdir(config["model_path"]):
        os.mkdir(config["model_path"])
    #加载训练数据
    train_data = load_data(config["train_data_path"], config)  # 加载训练测试集
    #加载模型
    model = TorchModel(config)
    # 标识是否使用gpu
    cuda_flag = torch.cuda.is_available()  # 检测机器是否可以使用gpu
    if cuda_flag:
        logger.info("gpu可以使用，迁移模型至gpu")
        model = model.cuda()
    #加载优化器
    optimizer = choose_optimizer(config, model)
    #加载效果测试类
    evaluator = Evaluator(config, model, logger)
    #训练
    for epoch in range(config["epoch"]):
        epoch += 1
        model.train()
        logger.info("epoch %d begin" % epoch)
        train_loss = []  # 创建一个列表，存放loss
        for index, batch_data in enumerate(train_data):  # 索引 加 值
            if cuda_flag:
                batch_data = [d.cuda() for d in batch_data]

            optimizer.zero_grad()
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            loss = model(input_ids, labels)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            if index % int(len(train_data) / 2) == 0:  # 这行代码是一个条件语句，用于检查 index 是否是 train_data 长度的一半的倍数。如果是，则执行下面的代码块。
                logger.info("batch loss %f" % loss)  # 当 index 是 train_data 长度的一半的倍数时，记录器 logger 记录一条 INFO 级别的日志，内容为 "batch loss %f" % loss。这里 %f 会被 loss 的值替换
        logger.info("epoch average loss: %f" % np.mean(train_loss))  #不论条件是否成立，这行代码都会执行。它记录了另一条 INFO 级别的日志，内容为 "epoch average loss: %f" % np.mean(train_loss)。这里 %f 会被 np.mean(train_loss) 的平均值替换。
        acc = evaluator.eval(epoch)  # 训练得到准确率
        # 每轮的结果都存下来
        # 在这直接加，注意文件名字
    # model_path = os.path.join(config["model_path"], "epoch_%d.pth" % epoch)
    # torch.save(model.state_dict(), model_path)  #保存模型权重
    return acc

if __name__ == "__main__":
    main(Config)

    # for model in ["cnn"]:
    #     Config["model_type"] = model
    #     print("最后一轮准确率：", main(Config), "当前配置：", Config["model_type"])

    #对比所有模型
    #中间日志可以关掉，避免输出过多信息
    # 超参数的网格搜索
    for model in ["gated_cnn"]: # ,"bert","lstm"
        Config["model_type"] = model
        for lr in [1e-3, 1e-4]:
            Config["learning_rate"] = lr
            for hidden_size in [128]:
                Config["hidden_size"] = hidden_size
                for batch_size in [64, 128]:
                    Config["batch_size"] = batch_size
                    for pooling_style in ["avg"]:  # "max"
                        Config["pooling_style"] = pooling_style
                        print("最后一轮准确率：", main(Config), "当前配置：", Config)  # 打开一个新的文件里面，把每轮的配置都记下来，输出csv
