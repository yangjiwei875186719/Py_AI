"""
加载模型
在🤗 Transformers库中，return_dict是from_pretrained方法中的一个参数，用于控制是否返回一个字典对象而不是多个输出。当return_dict=True时，模型的输出以字典的形式返回，其中包含模型的所有输出。这个参数通常用于方便地访问模型输出的不同部分，而不必通过索引来获取。
当return_dict=False时，模型将返回一个包含多个输出元素的元组，通常按照模型的输出顺序排列。这种方式可能需要额外的索引操作来获取所需的输出。
使用return_dict=True时，可以通过字典键来访问不同的模型输出，这样可以更容易地理解和处理模型的输出结果。这对于在训练、评估或推理过程中需要同时处理多个输出的情况特别有帮助。
"""
from transformers import BertModel,BertTokenizer
from config import Config
bert = BertModel.from_pretrained(Config["bert_path"], return_dict=False)
# print("bert:", bert)


# 加载bert词表
"""
text:
类型: Union[TextInput, PreTokenizedInput, EncodedInput]
描述: 输入的文本，可以是原始文本字符串、预分词后的文本或者编码后的文本。
text_pair:
类型: 可选的 Union[TextInput, PreTokenizedInput, EncodedInput]
描述: 可选的第二个句子，用于处理句对任务（例如文本分类中的文本对）。
add_special_tokens:
类型: 布尔值
描述: 是否添加特殊标记（如[CLS]和[SEP]）到编码序列中。
padding:
类型: Union[bool, str, PaddingStrategy]
描述: 是否对输入进行填充，可以是布尔值或填充策略，用于确保输入序列达到指定的最大长度。
truncation:
类型: Union[bool, str, TruncationStrategy]
描述: 是否对输入进行截断，可以是布尔值或截断策略，用于处理超过最大长度的输入。
max_length:
类型: 可选的整数
描述: 指定处理后的文本序列的最大长度，通常用于截断和填充操作。
stride:
类型: 整数
描述: 在截断时，指定截断窗口的滑动距离。
return_tensors:
类型: 可选的字符串或张量类型
描述: 指定返回的张量类型，如'pt'表示返回PyTorch张量。
**kwargs:
类型: 关键字参数
描述: 其余未列出的关键字参数将被传递给内部调用的方法。


用到的参数
add_special_tokens=False:去掉前缀
max_length=Config["max_length"]:截取文本最大长度
truncation=True:参数告诉tokenizer.encode方法在处理输入文本时要进行截断操作
padding="max_length:padding='max_length'参数告诉tokenizer.encode方法在处理输入文本时要进行填充操作，并且填充到指定的最大长度。这样可以确保输入序列达到指定的长度，以符合BERT模型对输入序列长度的要求
"""

def load_vocab(vocab_path):
    return BertTokenizer.from_pretrained(vocab_path)
tokenizer = load_vocab(Config["bert_path"])  # 这个是bert路径，不是bert的字典路径
print("tokenizer:",tokenizer)

vocab_index = tokenizer.encode("你好呀！",padding="max_length",max_length=Config["max_length"],truncation=True,) ##  ，
print("vocab_index:",vocab_index)