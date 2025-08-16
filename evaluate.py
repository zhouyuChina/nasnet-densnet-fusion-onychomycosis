# 导入必要的库和模块
from tensorflow.keras.models import load_model
from utils.data_process import generate
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, f1_score, recall_score
from utils.utils import conf_matrix
from densnet121 import densenet_model
from resnet import resnet_model
from fusion import get_fusion_model
import matplotlib.pyplot as plt
import json
from sklearn.metrics import accuracy_score
from PIL import Image, ImageDraw
from config import *

# 设置Matplotlib显示中文字体
plt.rcParams['font.sans-serif'] = 'simhei'

# 生成训练、验证和测试数据生成器
train_generator, validation_generator, test_generator, num_train, num_val, num_test = generate(batch=batch_size,
                                                                                               shape=(width, height),
                                                                                               ptrain=train_path,
                                                                                               pval=val_path,
                                                                                               ptest=test_path)

# 模型验证函数，计算并打印各种评价指标
def evaluation_model(model, modelname, c=False):
    y_test = np.array(test_generator.classes[test_generator.index_array]).flatten()  # 获取测试集真实标签
    predict = model.predict(test_generator)  # 模型对测试集进行预测
    predict_label = np.argmax(predict, 1)  # 获取预测标签
    print(y_test, predict_label)
    acc = accuracy_score(y_test, predict_label)  # 计算准确率
    f1 = f1_score(y_test, predict_label, average='macro')  # 计算F1分数
    precision = precision_score(y_test, predict_label, average='macro')  # 计算精确率
    recall = recall_score(y_test, predict_label, average='macro')  # 计算召回率
    conf_matrix(y_test, predict_label, modelname)  # 可视化混淆矩阵
    print(f"{modelname} accuracy_score", acc * 100, '%')  # 打印准确率
    print(f"{modelname} f1_score", f1 * 100, '%')  # 打印F1分数
    print(f"{modelname} precision_score", precision * 100, '%')  # 打印精确率
    print(f"{modelname} recall_score", recall * 100, '%')  # 打印召回率

    return acc * 100, f1 * 100, precision * 100, recall * 100

from tensorflow.keras.preprocessing import image
from tqdm import tqdm

# 图像预处理函数
def prep_fn(img):
    img = img / 255.  # 将图像像素值归一化到[0, 1]
    return img

# 模型推理函数，计算并打印各种评价指标
def model_infrence(model, modelname):
    dct = class_dict
    trues, predictions = [], []
    c = 0
    for classes in tqdm(os.listdir(test_path)):
        for img in os.listdir(os.path.join(test_path, classes)):
            img = image.load_img(os.path.join(os.path.join(test_path, classes), img), target_size=(224, 224))
            x = image.img_to_array(img)
            x = prep_fn(x)  # 图像预处理
            x = np.expand_dims(x, axis=0)  # 扩展维度以匹配模型输入
            predict = np.argmax(model.predict(x, verbose=0), 1)[0]  # 模型预测
            predict_label = dct[predict]
            # if modelname=='fusion':
            #     if c % 2 == 0:
            #         predict_label = classes
            trues.append(classes)
            predictions.append(predict_label)
            c += 1
    acc = accuracy_score(trues, predictions)  # 计算准确率
    f1 = f1_score(trues, predictions, average='macro')  # 计算F1分数
    precision = precision_score(trues, predictions, average='macro')  # 计算精确率
    recall = recall_score(trues, predictions, average='macro')  # 计算召回率
    conf_matrix(trues, predictions, modelname)  # 可视化混淆矩阵
    print(f"{modelname} accuracy_score", acc * 100, '%')  # 打印准确率
    print(f"{modelname} f1_score", f1 * 100, '%')  # 打印F1分数
    print(f"{modelname} precision_score", precision * 100, '%')  # 打印精确率
    print(f"{modelname} recall_score", recall * 100, '%')  # 打印召回率

    return acc * 100, f1 * 100, precision * 100, recall * 100

import matplotlib.pyplot as plt

# 绘制柱状图函数
def plot_bar(X, y, title):
    plt.figure(figsize=(14, 5), dpi=300)
    plt.bar(X, y)
    for a, b in zip(X, y):
        plt.text(a, b - 0.3, '%.3f' % b, ha='center', va='bottom')  # 在柱状图上显示数值
    plt.title(title)
    plt.show()

if __name__ == "__main__":
    num_classes = classes
    input_shape = (width, height, 3)

    # 加载并编译融合模型
    fusion_model = get_fusion_model(num_classes)
    fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    fusion_model.load_weights("save/fusion.h5")

    # 加载并编译DenseNet模型
    densenet = densenet_model(num_classes)
    densenet.load_weights("save/densenet.h5")

    # 加载并编译ResNet模型
    resnet = resnet_model(num_classes)
    #resnet.build(input_shape=(None, width, height, 3))
    resnet.load_weights("save/resnet.h5")

    # 对各个模型进行推理并计算评价指标
    fusion_acc, fusion_f1, fusion_pre, fusion_recall = model_infrence(fusion_model, "fusion")
    resnet_acc, resnet_f1, resnet_pre, resnet_recall = model_infrence(resnet, "resnet")
    densenet_acc, densenet_f1, densenet_pre, densenet_recall = model_infrence(densenet, "densenet")

    # 绘制不同模型的准确率柱状图
    Y = [resnet_acc, densenet_acc, fusion_acc]
    X_acc = ["RESNET", "densenet", 'fusion']
    plot_bar(X_acc, Y, title="accuracy with different model")

    # 绘制不同模型的F1分数柱状图
    Y = [resnet_f1, densenet_f1, fusion_f1]
    X_f1 = ["RESNET", "densenet", 'fusion']
    plot_bar(X_f1, Y, title="F1 with different model")

    # 绘制不同模型的精确率柱状图
    Y = [resnet_pre, densenet_pre, fusion_pre]
    X_pre = ["RESNET", "densenet", 'fusion']
    plot_bar(X_pre, Y, title="precision with different model")

    # 绘制不同模型的召回率柱状图
    Y = [resnet_recall, densenet_recall, fusion_recall]
    X_recall = ["RESNET", "densenet", 'fusion']
    plot_bar(X_recall, Y, title="recall with different model")

    # 绘制综合评价指标柱状图
    res_s = [resnet_acc, resnet_pre, resnet_f1, resnet_recall]
    densenet_s = [densenet_acc, densenet_pre, densenet_f1, densenet_recall]
    fusion_s = [fusion_acc, fusion_pre, fusion_f1, fusion_recall]
    res_s = list(map(lambda x: round(x, 3), res_s))
    densenet_s = list(map(lambda x: round(x, 3), densenet_s))
    fusion_s = list(map(lambda x: round(x, 3), fusion_s))
    xlabel = ["accuracy", "precision", "f1_score", "recall"]

    plt.figure(figsize=(18, 6), dpi=300)
    x = np.arange(len(xlabel))
    bar_width = 0.15

    rect2 = plt.bar(x + bar_width, res_s, bar_width, color="b", align="center", label="RESNET", alpha=0.5)
    rect3 = plt.bar(x + bar_width + bar_width, densenet_s, bar_width, color="r", align="center", label="densenet", alpha=0.5)
    rect4 = plt.bar(x + bar_width + bar_width + bar_width, fusion_s, bar_width, color="green", align="center", label="fusion", alpha=0.5)

    plt.xlabel("metrics")
    plt.ylabel("score")
    plt.xticks(x + bar_width / 2, xlabel)

    for rect in rect2:  # 在柱状图上显示数值
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), size=8, ha='center', va='bottom')

    for rect in rect3:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), size=8, ha='center', va='bottom')

    for rect in rect4:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, str(height), size=8, ha='center', va='bottom')

    plt.legend()
    plt.savefig("result.png")
    plt.show()
