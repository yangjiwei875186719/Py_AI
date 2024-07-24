import math
from collections import defaultdict


class NgramLanguageModel:
    def __init__(self, corpus=None, n=3):  # corpus截好的词， n 几元组
        self.n = n
        self.sep = "_"     # 用来分割两个词，没有实际含义，只要是字典里不存在的符号都可以
        self.sos = "<sos>"    #start of sentence，句子开始的标识符
        self.eos = "<eos>"    #end of sentence，句子结束的标识符
        self.unk_prob = 1e-5  #给unk分配一个比较小的概率值，避免集外词概率为0
        self.fix_backoff_prob = 0.4  #使用固定的回退概率
        self.ngram_count_dict = dict((x + 1, defaultdict(int)) for x in range(n)) # 字典 # range(n) 生成了一个从 0 到 n-1 的数字序列。defaultdict(int)如果取不到默认值为整数0
        self.ngram_count_prob_dict = dict((x + 1, defaultdict(int)) for x in range(n)) # 新的字典
        self.ngram_count(corpus)
        self.calc_ngram_prob()

    #将文本切分成词或字或token
    def sentence_segment(self, sentence):
        return sentence.split()  # 文本空格切分
        #return jieba.lcut(sentence)

    #统计ngram的数量
    def ngram_count(self, corpus):
        for sentence in corpus:  # 文本循环 一行一行的打印循环
            word_lists = self.sentence_segment(sentence)
            word_lists = [self.sos] + word_lists + [self.eos]  #前后补充开始符和结尾符，一行文本加首尾符号
            # print("word_lists",word_lists)
            for window_size in range(1, self.n + 1):           #按不同窗长扫描文本 1 2,3 ，1组 2组 3组都要算一遍
                for index, word in enumerate(word_lists):  # 索引 词
                    #取到末尾时窗口长度会小于指定的gram，跳过那几个
                    if len(word_lists[index:index + window_size]) != window_size:
                        continue
                    #用分隔符连接word形成一个ngram用于存储
                    ngram = self.sep.join(word_lists[index:index + window_size]) # 获取指定索引的值拼接 拼接符号是seq
                    # print("ngram:",ngram)  # 例如： ngram: d_b_c
                    self.ngram_count_dict[window_size][ngram] += 1  # 字典窗口长度作为key， ngram作为value 已经汇总过了
        #计算总词数，后续用于计算一阶ngram概率
        self.ngram_count_dict[0] = sum(self.ngram_count_dict[1].values())
        print("self.ngram_count_dict", self.ngram_count_dict)  # 打印字典，例如： {1: defaultdict(<class 'int'>, {'<sos>': 0.13114754098360656, 'a': 0.08196721311475409, 'b': 0.13114754098360656, 'c': 0.09836065573770492, 'd': 0.11475409836065574, '<eos>': 0.13114754098360656, 'e': 0.14754098360655737, 'f': 0.09836065573770492, 'g': 0.06557377049180328})
        # print("self.ngram_count_dict0",self.ngram_count_dict[0])  # 获取字典 key为0 的value值
        #
        # print("self.ngram_count_dict1", self.ngram_count_dict[1])
        # print("self.ngram_count_dict2", self.ngram_count_dict[2])
        # print("self.ngram_count_dict3", self.ngram_count_dict[3])
        return

    #计算ngram概率
    def calc_ngram_prob(self):
        for window_size in range(1, self.n + 1):
            for ngram, count in self.ngram_count_dict[window_size].items(): # 对窗口进行遍历,也就是key遍历  ，count 是对出现次数的累积或聚合。
                # print("ngram:",ngram,"count:",count)  # 例如 ngram: b_c_d count: 2
                if window_size > 1:
                    # 以sep为切分，最后一位不取，其他的都取
                    ngram_splits = ngram.split(self.sep)              #ngram        :a b c  例如p(今天 天气 不错)
                    ngram_prefix = self.sep.join(ngram_splits[:-1])   #ngram_prefix :a b    例如p(今天 不错)

                    ngram_prefix_count = self.ngram_count_dict[window_size - 1][ngram_prefix] #Count(a,b)  # 降一个窗口的长度，找p(今天 不错) 的数量
                    print("ngram_prefix_count",ngram_prefix_count)
                else:
                    ngram_prefix_count = self.ngram_count_dict[0]     #count(total word) # 如果
                # word = ngram_splits[-1]
                # self.ngram_count_prob_dict[word + "|" + ngram_prefix] = count / ngram_prefix_count
                self.ngram_count_prob_dict[window_size][ngram] = count / ngram_prefix_count
            print("self.ngram_count_prob_dict:",self.ngram_count_prob_dict)
        return

    # 获取ngram概率，其中用到了回退平滑，回退概率采取固定值
    def get_ngram_prob(self, ngram):
        n = len(ngram.split(self.sep))
        if ngram in self.ngram_count_prob_dict[n]:
            # 尝试直接取出概率
            return self.ngram_count_prob_dict[n][ngram]
        elif n == 1:  # 如果真回退到一阶，就给个默认概率
            # 一阶gram查找不到，说明是集外词，不做回退
            return self.unk_prob  #
        else:
            # 高于一阶的可以回退
            ngram = self.sep.join(ngram.split(self.sep)[1:])  # abc 回退  取bc
            return self.fix_backoff_prob * self.get_ngram_prob(ngram)  # 固定回头概率 * 当前方法（如果当前方法还没有找到，就继续回退）


    #回退法预测句子概率 困惑率
    def calc_sentence_ppl(self, sentence):
        word_list = self.sentence_segment(sentence)
        word_list = [self.sos] + word_list + [self.eos]
        sentence_prob = 0
        for index, word in enumerate(word_list):  # 对词进行遍历 index索引,word值
            ngram = self.sep.join(word_list[max(0, index - self.n + 1):index + 1])
            prob = self.get_ngram_prob(ngram)
            # print(ngram, prob)
            sentence_prob += math.log(prob)
        return 2 ** (sentence_prob * (-1 / len(word_list)))  # ppl困惑率 2 的 -1/n * log累加 次方



if __name__ == "__main__":
    corpus = open("sample.txt", encoding="utf8").readlines()
    lm = NgramLanguageModel(corpus, 3)
    print("词总数:", lm.ngram_count_dict[0])
    print("lm.ngram_count_prob_dict:",lm.ngram_count_prob_dict)
    print(lm.calc_sentence_ppl("c d b d b"))
