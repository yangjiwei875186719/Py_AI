# -%- coding:utf-8 -*-
import json

import torch
from transformers import BertTokenizer
from model import TorchModel
from collections import defaultdict
import re
class SentenceEntity:
    def __init__(self,config,model_path):
        self.config = config
        self.schema = self.load_schema(config["schema_path"])
        self.schema_list = ['LOCATION', 'ORGANIZATION', 'PERSON', 'TIME']
        self.index_to_sign = dict((y,x) for x, y in self.schema.items())
        self.tokenizer = BertTokenizer.from_pretrained(config["bert_path"])
        self.vocab = self.load_vocab(config["bert_vocab_path"])
        self.model = TorchModel(config)
        self.model.load_state_dict(torch.load(model_path))

        self.model.eval()
        print("模型加载完毕！")

    def load_schema(self,path):
        with open(path,encoding="utf-8") as f:
            schema = json.load(f)
            self.config["class_num"] = len(schema)
        return schema
    # 加载字表或词表
    def load_vocab(self, vocab_path):
        token_dict = {}
        with open(vocab_path, encoding="utf8") as f:
            for index, line in enumerate(f):
                token = line.strip()
                token_dict[token] = index + 1  # 0留给padding位置，所以从1开始
        self.config["vocab_size"] = len(token_dict)
        return token_dict
    def predict(self,sentence):
        print("sentence",sentence)
        sentences = []
        for char in sentence:
            sentences.append(char.strip())
        print("sentences",sentences)
        input_id = self.tokenizer.encode(sentences, max_length=self.config["max_length"], truncation=True,
                                          padding='max_length')
        # input_id = self.tokenizer.encode(sentence, add_special_tokens=False, max_length=self.config['max_length'],
        #                                  padding='max_length',
        #                                  truncation=True)
        print("input_id",input_id)
        with torch.no_grad():
            input_tensor = torch.LongTensor([input_id])  # 转换为二维张量
            pred = self.model(input_tensor)[0]
            print("pred",pred)
            pred = torch.argmax(pred,dim = -1)
            print(pred)
            # pred = pred.detach().cpu().tolist()
        #     pred = "".join([str(c) for c in pred[:len(sentence)]])
        # res = defaultdict(list)
        # for locations in re.finditer("(04+)", pred):
        #     s, e = locations.span()
        #     res['LOCATION'].append(sentence[s: e])
        # for locations in re.finditer("(15+)", pred):
        #     s, e = locations.span()
        #     res['ORGANIZATION'].append(sentence[s: e])
        # for locations in re.finditer("(26+)", pred):
        #     s, e = locations.span()
        #     res['PERSON'].append(sentence[s: e])
        # for locations in re.finditer("(37+)", pred):
        #     s, e = locations.span()
        #     res['TIME'].append(sentence[s: e])
        # return res
if __name__ == '__main__':
    from config import Config
    sl = SentenceEntity(Config,"model_output/epoch_50.pth")
    sentence = "2024年3月25日，中国在南非建立了驻使馆，如果有什么需要，中华人民共和国会保护我们的"
    res = sl.predict(sentence)
    print(res)