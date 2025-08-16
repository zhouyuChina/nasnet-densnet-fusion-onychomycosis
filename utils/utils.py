"""
一些工具代码
"""
import matplotlib.pyplot as plt
from config import *
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import pandas as pd
from tensorflow.keras import backend as K
import tensorflow as tf

plt.rcParams['font.sans-serif'] = ['simhei']


def plot_history(history, modelname):
    """定义画训练日志的函数"""
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    plt.figure(figsize=(12, 4), dpi=300)
    plt.subplot(1, 2, 1)
    plt.plot(range(len(loss)), loss, label='loss')
    plt.plot(range(len(loss)), val_loss, label='val_loss')
    plt.grid()
    plt.legend()
    plt.title(f"{modelname}  loss")
    plt.subplot(1, 2, 2)
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(range(len(acc)), acc, label='accuracy')
    plt.plot(range(len(acc)), val_acc, label='val_accuracy')
    plt.grid()
    plt.legend()
    plt.title(f"{modelname}  accuracy")
    check_path((os.path.join(logs_path, modelname)))
    plt.savefig(os.path.join(os.path.join(logs_path, modelname), 'train_log.jpg'))
    plt.show()


def conf_matrix(y_true, y_predict, modelname):
    """画混淆矩阵的函数"""
    plt.figure(figsize=(12, 8), dpi=300)
    matrix = pd.DataFrame(confusion_matrix(y_true, y_predict), columns=class_dict)
    matrix.index = matrix.columns
    sns.heatmap(matrix, annot=True, fmt='.20g')
    plt.title(f"{modelname} confusion_matrix")
    plt.savefig(os.path.join(os.path.join(logs_path, modelname), 'confusion_matrix.png'))
    plt.xticks(rotation=45)
    plt.show()


def classifaction_report_csv(report):
    """得到各个类别的分类指标-P R f1"""
    report_data = []
    lines = report.split('\n')
    for line in lines[2:-5]:
        row = {}
        row_datas = line.split('      ')

        row_datas = list(map(lambda x: x.strip(), row_datas))
        row_data = []
        for r in row_datas:
            if not r == '':
                row_data.append(r)

        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    df = pd.DataFrame.from_dict(report_data)
    return df


def plot_report(report_df, modelname):
    """把上面的数据可视化为热力图"""
    # 可视化
    report_df.iloc[:, 1:] = report_df.iloc[:, 1:].astype(float)
    report_df.index = report_df['class']
    report_df = report_df.drop(['class'], axis=1)
    plt.figure(figsize=(12, 4), dpi=300)
    ### annot是表示显示方块代表的数值出来 cmap颜色
    sns.heatmap(report_df.iloc[:, :-1], annot=True, cmap="YlGnBu")
    plt.title(f"{modelname} classification heatmap")
    plt.savefig(os.path.join(os.path.join(logs_path, modelname), "classification_heatmap.png"))
    plt.show()


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.ones_like(y_pred))
        focal_loss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(
            K.epsilon() + pt_1)) - K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0 + K.epsilon()))
        return focal_loss

    return focal_loss_fixed


from tensorflow.keras.callbacks import Callback


class WarmupExponentialDecay(Callback):
    def __init__(self, lr_base=0.0002, lr_min=0.0, decay=0, warmup_epochs=0):
        self.num_passed_batchs = 0  # 一个计数器
        self.warmup_epochs = warmup_epochs
        self.lr = lr_base  # learning_rate_base
        self.lr_min = lr_min  # 最小的起始学习率,此代码尚未实现
        self.decay = decay  # 指数衰减率
        self.steps_per_epoch = 0  # 也是一个计数器

    def on_batch_begin(self, batch, logs=None):
        # params是模型自动传递给Callback的一些参数
        if self.steps_per_epoch == 0:
            # 防止跑验证集的时候呗更改了
            if self.params['steps'] == None:
                self.steps_per_epoch = np.ceil(1. * self.params['samples'] / self.params['batch_size'])
            else:
                self.steps_per_epoch = self.params['steps']
        if self.num_passed_batchs < self.steps_per_epoch * self.warmup_epochs:
            K.set_value(self.model.optimizer.lr,
                        self.lr * (self.num_passed_batchs + 1) / self.steps_per_epoch / self.warmup_epochs)
        else:
            K.set_value(self.model.optimizer.lr,
                        self.lr * ((1 - self.decay) ** (
                                    self.num_passed_batchs - self.steps_per_epoch * self.warmup_epochs)))
        self.num_passed_batchs += 1

    def on_epoch_begin(self, epoch, logs=None):
        # 用来输出学习率的,可以删除
        print("learning_rate:", K.get_value(self.model.optimizer.lr))
