"""
配置文件
包括输入图像维度，batch_size, learning_rate, 训练比例等等
"""
import os

width = 224
height = 224  # 输入维度
batch_size = 4  # batchsize
learning_rate = 1e-4  # 学习率
train_ratio = 0.7  # 训练比例
test_ratio = 0.2
classes = 2  # 类别数
epochs = 500  # 迭代次数

split_path = "./split_dataset"
train_path = "./split_dataset/train"
val_path = "./split_dataset/valid"
test_path = "./split_dataset/test"
save_model = './save/'
logs_path = "./logs"
a = {'normal': 0, 'onychomycosis': 1}
class_dict = list(a.keys())

import os


def check_path(path):
    if not os.path.exists(path):
        os.mkdir(path)


check_path(save_model)
check_path(logs_path)
check_path(split_path)
check_path(train_path)
check_path(test_path)
check_path(val_path)
