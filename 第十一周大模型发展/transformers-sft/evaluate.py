# -*- coding: utf-8 -*-
from loader import load_data,load_vocab
from collections import defaultdict
import model
class Evaluator:
    def __init__(self,config,model,logger,tokenizer):
        self.config = config
        self.model = model
        self.logger =logger
        self.tokenizer = tokenizer
        # 加载测试训练集
        self.valid_data = load_data(config["valid_data_path"],config,logger,shuffle=False)
        self.reverse_vocab = {v: k for k, v in self.tokenizer.get_vocab().items()}
        # for key ,value in self.reverse_vocab.items():
            # print(f"key{key},value{value}")
        # print("self.reverse_vocab", self.reverse_vocab["11"])'
    def eval(self,epoch):
        self.logger.info(f"开始测试第{epoch}轮模型效果：")
        self.model.eval()
        self.model.cpu()
        self.stats_dict = defaultdict(int) # 用于存储测试结果
        for index,batch_data in enumerate(self.valid_data):
            input_seqs,output_seqs = batch_data
            for input_seq in input_seqs:
                generate = self.tokenizer.decode_seq(input_seq)
                print("generate",generate)


    def decode_seq(self, seq):
        return "".join([self.reverse_vocab[int(idx)] for idx in seq])




if __name__ == '__main__':
    from config import Config
    tokenizer = load_vocab(Config["bert_path"])
