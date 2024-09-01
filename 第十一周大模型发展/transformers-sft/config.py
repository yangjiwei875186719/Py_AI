# -*- coding: utf-8 -*-

"""
配置参数信息
"""
import os
import torch

Config = {
    "train_data_path": r"./data/sample_data.json",
    "valid_data_path": r"./data/sample_data.json",
    "bert_path":r"D:\appdev\PyProject\Py_AI\第六周预训练模型\bert-base-chinese",
    "input_max_length": 30,
    "output_max_length": 120,
    "max_length": 100,
    "optimizer":"adam",
    "model_path": "output",

    "epoch": 200,
    "batch_size": 32,
    "learning_rate":1e-3,
    "seed":42,
    "beam_size":5

    }

