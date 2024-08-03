# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "../data/schema.json",
    "train_data_path": "../data/train.json",
    "valid_data_path": "../data/valid.json",
    "pretrain_model_path":r"F:\Desktop\work_space\pretrain_models\bert-base-chinese",
    "vocab_path":r"F:\Desktop\work_space\pretrain_models\bert-base-chinese\vocab.txt",
    "max_length": 20,
    "hidden_size": 256,
    "epoch": 10,
    "batch_size": 128,
    "epoch_data_size": 10000,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
}