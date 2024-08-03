# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):  # 构造函数初始化 Evaluator 类的实例
        self.config = config   # 将传入的配置信息存储在类的实例中。
        self.model = model
        self.logger = logger
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False) # 加载测试集
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.model.eval()
        self.stats_dict = {"correct": 0, "wrong": 0}  # 清空上一轮结果
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_ids, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():   # 禁用梯度计算，进行模型的推断，得到预测结果 pred_results。
                pred_results = self.model(input_ids) #不输入labels，使用模型当前参数进行预测
            self.write_stats(labels, pred_results)
        acc = self.show_stats()
        return acc
    """计算预测结果准确性的函数"""
    def write_stats(self, labels, pred_results):  # 它接受两个参数 labels 和 pred_results，分别代表真实标签和预测结果
        assert len(labels) == len(pred_results)  # 确保真实标签和预测结果的长度相等，以便进行比较。
        for true_label, pred_label in zip(labels, pred_results):   # 函数首先确保 labels 和 pred_results 的长度相等，然后通过 zip 函数同时迭代它们。
            pred_label = torch.argmax(pred_label)   # 在每次迭代中，代码将预测结果转换为概率最大的类别（使用 torch.argmax），然后将真实标签和预测标签进行比较。如果它们相等，表示预测正确，将 self.stats_dict 中的 "correct" 键对应的值加一；否则，将 "wrong" 键对应的值加一。
            if int(true_label) == int(pred_label):  # 这是一个条件语句，用于比较真实标签和预测标签是否相等。如果相等，表示预测正确，将 self.stats_dict 中的 "correct" 键对应的值加一；否则，将 "wrong" 键对应的值加一。
                self.stats_dict["correct"] += 1  #  存入到self.stats_dict
            else:
                self.stats_dict["wrong"] += 1
        return
    """ 预测准确率，用于展示模型的预测统计信息。函数从 self.stats_dict 中获取“correct”和“wrong”键对应的值，分别表示正确的预测数量和错误的预测数量。"""
    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))  # 正确预测条目数量和错误预测条目数量的总和。
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))  # 正确预测条目数量和错误预测条目数量。
        self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))  # 预测准确率：正确预测条目数量除以总预测条目数量（正确预测条目数量加上错误预测条目数量
        self.logger.info("--------------------")
        return correct / (correct + wrong)
