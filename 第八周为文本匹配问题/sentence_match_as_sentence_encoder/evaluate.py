# -*- coding: utf-8 -*-
import torch
from loader import load_data

"""
模型效果测试
"""

class Evaluator:
    def __init__(self, config, model, logger):
        self.config = config
        self.model = model
        self.logger = logger

        # load_data中的if isinstance(line, dict):判断是否是字段格式的，不是字典格式的就是test,如果自动匹配是执行测试集加载还是训练集加载
        self.valid_data = load_data(config["valid_data_path"], config, shuffle=False)  # 加载测试集
        # 由于效果测试需要训练集当做知识库，再次加载训练集。
        # 事实上可以通过传参把前面加载的训练集传进来更合理，但是为了主流程代码改动量小，在这里重新加载一遍
        self.train_data = load_data(config["train_data_path"], config)
        self.stats_dict = {"correct":0, "wrong":0}  #用于存储测试结果

    #将知识库中的问题向量化，为匹配做准备
    #每轮训练的模型参数不一样，生成的向量也不一样，所以需要每轮测试都重新进行向量化
    def knwb_to_vector(self):
        self.question_index_to_standard_question_index = {}
        self.question_ids = []
        # self.question_ids：这是一个列表，用于存储训练集中所有问题的ID（或索引）。这些ID将被用于后续生成对应的向量。
        # self.train_data.dataset.knwb 来访问训练集中的知识库。这里假设 knwb 是一个字典，其键是标准问题索引，值是与该标准问题相关联的问题ID列表
        # 方法将每个问题ID添加到 self.question_ids 列表中，并在 self.question_index_to_standard_question_index 字典中记录当前索引（即 len(self.question_ids)）与标准问题索引之间的映射关系。
        for standard_question_index, question_ids in self.train_data.dataset.knwb.items():
            for question_id in question_ids:  # 遍历训练集value的值
                print("standard_question_index:",standard_question_index,"question_id:",question_id)
                print("len(self.question_ids):",len(self.question_ids))
                #记录问题编号到标准问题标号的映射，用来确认答案是否正确
                self.question_index_to_standard_question_index[len(self.question_ids)] = standard_question_index
                self.question_ids.append(question_id)
        with torch.no_grad():  # torch.no_grad()：这是一个好的做法，特别是当你不需要计算梯度（比如在推理或评估模型时）时。它告诉PyTorch不要追踪对当前操作（及之后的操作）的梯度，这可以节省内存并加速计算
            question_matrixs = torch.stack(self.question_ids, dim=0)  # 你通过 torch.stack(self.question_ids, dim=0) 将 self.question_ids（假设是一个tensor列表）沿着第一个维度（dim=0）堆叠起来，形成一个新的tensor。这里确保 self.question_ids 中的每个tensor都具有相同的形状，否则 torch.stack 会报错
            if torch.cuda.is_available():  # 你检查了CUDA是否可用，并在可用时将 question_matrixs 移动到GPU上。这是进行GPU加速计算的标准做法。
                question_matrixs = question_matrixs.cuda()
            self.knwb_vectors = self.model(question_matrixs)  # 过模型形成向量  通过模型获取向量：你通过 self.model(question_matrixs) 将 question_matrixs 传递给模型，并获取输出的向量（或向量矩阵）。这里假设 self.model 已经正确定义，并且可以接受这种类型的输入
            #将所有向量都作归一化 v / |v|
            # 你使用 torch.nn.functional.normalize 对输出的向量进行归一化处理，这通常是有益的，因为它可以帮助在后续处理中（如计算相似度）减少数值问题。dim=-1 指定了在最后一个维度上进行归一化，这通常是针对向量的每个元素进行的（即对每个向量的长度进行归一化）
            self.knwb_vectors = torch.nn.functional.normalize(self.knwb_vectors, dim=-1)
        return

    def eval(self, epoch):
        self.logger.info("开始测试第%d轮模型效果：" % epoch)
        self.stats_dict = {"correct":0, "wrong":0}  #清空前一轮的测试结果
        self.model.eval()
        self.knwb_to_vector()
        for index, batch_data in enumerate(self.valid_data):
            if torch.cuda.is_available():
                batch_data = [d.cuda() for d in batch_data]
            input_id, labels = batch_data   #输入变化时这里需要修改，比如多输入，多输出的情况
            with torch.no_grad():
                test_question_vectors = self.model(input_id) #不输入labels，使用模型当前参数进行预测
            self.write_stats(test_question_vectors, labels)
        self.show_stats()
        return

    def write_stats(self, test_question_vectors, labels):
        assert len(labels) == len(test_question_vectors)  # 确保 self.knwb_vectors 和 self.question_index_to_standard_question_index 已被正确初始化：在调用 write_stats 方法之前，需要确保 self.knwb_vectors 包含了知识库中所有问题的向量表示，并且 self.question_index_to_standard_question_index 是一个映射，用于将内部问题索引转换为标准问题编号
        for test_question_vector, label in zip(test_question_vectors, labels):
            # 通过一次矩阵乘法，计算输入问题和知识库中所有问题的相似度
            # test_question_vector shape [vec_size]   knwb_vectors shape = [n, vec_size]
            # .unsqueeze() 和 .T 来调整矩阵维度：你的代码中使用 .unsqueeze(0) 来给 test_question_vector 增加一个批次维度，然后使用 .T 来转置 self.knwb_vectors，以便进行矩阵乘法。这是正确的，但请确保 self.knwb_vectors 的形状是 [n, vec_size]，其中 n 是知识库中问题的数量，vec_size 是每个问题向量的维度。
            res = torch.mm(test_question_vector.unsqueeze(0), self.knwb_vectors.T)
            # torch.argmax 的结果：你使用 int(torch.argmax(res.squeeze())) 来获取相似度最高的索引。这是正确的，但请注意，如果 res 很小或接近零，则最大值的索引可能不稳定（尽管在大多数情况下这不是问题）
            hit_index = int(torch.argmax(res.squeeze())) #命中问题标号
            hit_index = self.question_index_to_standard_question_index[hit_index] #转化成标准问编号
            if int(hit_index) == int(label):
                # 初始化 self.stats_dict：在 write_stats 方法中，你假设 self.stats_dict 已经被初始化并包含了 "correct" 和 "wrong" 这两个键。确保在类的某个地方（可能是 __init__ 方法中）已经进行了这样的初始化。
                self.stats_dict["correct"] += 1
            else:
                self.stats_dict["wrong"] += 1
        # 你的函数目前没有返回值（除了隐式的 None）。是为了计算self.stats_dict的 准确率和错误率
        return
    # 你提供的 show_stats 方法用于展示模型的预测统计信息，包括预测总量、预测正确的条目数、预测错误的条目数以及预测准确率。这个方法看起来是合理的，但有一点小改进可以在处理除法时增加，即当 correct + wrong 为0时（即没有预测结果时），直接除以0会导致运行时错误。为了避免这种情况，你可以添加一个条件判断来优雅地处理这种边界情况。
    def show_stats(self):
        correct = self.stats_dict["correct"]
        wrong = self.stats_dict["wrong"]
        total = correct + wrong
        self.logger.info("预测集合条目总量：%d" % (correct +wrong))
        self.logger.info("预测正确条目：%d，预测错误条目：%d" % (correct, wrong))
        # 添加条件判断以避免除以0的错误
        if total > 0:
            accuracy = correct / total
            self.logger.info("预测准确率：%f" % accuracy)
        else:
            self.logger.info("预测准确率：无法计算（因为没有预测结果）")

        self.logger.info("--------------------")
        return
        # self.logger.info("预测准确率：%f" % (correct / (correct + wrong)))
        # self.logger.info("--------------------")
        # return
