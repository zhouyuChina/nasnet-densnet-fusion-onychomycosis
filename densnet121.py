# 导入必要的库和模块
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Sequential, backend
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, BatchNormalization, Activation, \
    GlobalAveragePooling2D
from tensorflow.keras.layers import Concatenate, Lambda, Input, ZeroPadding2D, AveragePooling2D
from config import *  # 导入配置文件中的参数
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from utils.utils import *  # 导入工具函数
from utils.data_process import generate  # 导入数据生成器函数
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model

# 定义卷积块
def conv_block(x, nb_filter, dropout_rate=None, name=None):
    inter_channel = nb_filter * 4  # 中间层通道数

    # 1x1 卷积
    x = BatchNormalization(epsilon=1.1e-5, axis=3, name=name + '_bn1')(x)  # 批归一化
    x = Activation('relu', name=name + '_relu1')(x)  # 激活函数
    x = Conv2D(inter_channel, 1, 1, name=name + '_conv1', use_bias=False)(x)  # 1x1 卷积

    if dropout_rate:
        x = Dropout(dropout_rate)(x)  # Dropout

    # 3x3 卷积
    x = BatchNormalization(epsilon=1.1e-5, axis=3, name=name + '_bn2')(x)  # 批归一化
    x = Activation('relu', name=name + '_relu2')(x)  # 激活函数
    x = ZeroPadding2D((1, 1), name=name + '_zeropadding2')(x)  # 零填充
    x = Conv2D(nb_filter, 3, 1, name=name + '_conv2', use_bias=False)(x)  # 3x3 卷积

    if dropout_rate:
        x = Dropout(dropout_rate)(x)  # Dropout

    return x  # 返回卷积块的输出


# 定义密集块
def dense_block(x, stage, nb_layers, nb_filter, growth_rate, dropout_rate=None,
                grow_nb_filters=True, name=None):
    concat_feat = x  # 存储最后一层的输出

    for i in range(nb_layers):
        branch = i + 1  # 分支编号
        x = conv_block(concat_feat, growth_rate, dropout_rate,
                       name=name + str(stage) + '_block' + str(branch))  # 使用相同的growth_rate
        concat_feat = Concatenate(axis=3, name=name + str(stage) + '_block' + str(branch))([concat_feat, x])  # 连接输出

        if grow_nb_filters:
            nb_filter += growth_rate  # 更新通道数

    return concat_feat, nb_filter  # 返回密集块的输出和更新后的通道数


# 定义过渡层
def transition_block(x, stage, nb_filter, compression=1.0, dropout_rate=None, name=None):
    x = BatchNormalization(epsilon=1.1e-5, axis=3, name=name + str(stage) + '_bn')(x)  # 批归一化
    x = Activation('relu', name=name + str(stage) + '_relu')(x)  # 激活函数
    x = Conv2D(int(nb_filter * compression), 1, 1, name=name + str(stage) + '_conv', use_bias=False)(x)  # 1x1 卷积

    if dropout_rate:
        x = Dropout(dropout_rate)(x)  # Dropout

    x = AveragePooling2D((2, 2), strides=(2, 2), name=name + str(stage) + '_pooling2d')(x)  # 平均池化

    return x  # 返回过渡层的输出


# 定义DenseNet模型
def DenseNet(nb_dense_block=4, growth_rate=32, nb_filter=64, reduction=0.0, dropout_rate=0.0, weight_decay=1e-4,
             classes=1000, weights_path=None):
    compression = 1.0 - reduction  # 压缩率
    nb_filter = 64  # 初始通道数
    nb_layers = [6, 12, 24, 16]  # 对应DenseNet-121的层数

    img_input = Input(shape=(width, height, 3))  # 输入层

    # 初始卷积层
    x = ZeroPadding2D((3, 3), name='conv1_zeropadding')(img_input)  # 零填充
    x = Conv2D(nb_filter, 7, 2, name='conv1', use_bias=False)(x)  # 7x7 卷积
    x = BatchNormalization(epsilon=1.1e-5, axis=3, name='conv1_bn')(x)  # 批归一化
    x = Activation('relu', name='relu1')(x)  # 激活函数
    x = ZeroPadding2D((1, 1), name='pool1_zeropadding')(x)  # 零填充
    x = MaxPooling2D((3, 3), strides=(2, 2), name='pool1')(x)  # 最大池化

    # 密集块和过渡层
    for block_id in range(nb_dense_block - 1):
        stage = block_id + 2  # 从2开始
        x, nb_filter = dense_block(x, stage, nb_layers[block_id], nb_filter, growth_rate,
                                   dropout_rate=dropout_rate, name='Dense')

        x = transition_block(x, stage, nb_filter, compression=compression, dropout_rate=dropout_rate, name='Trans')
        nb_filter *= compression  # 更新通道数

    final_stage = stage + 1  # 最后一个密集块的阶段编号
    x, nb_filter = dense_block(x, final_stage, nb_layers[-1], nb_filter, growth_rate,
                               dropout_rate=dropout_rate, name='Dense')

    # 顶层
    x = BatchNormalization(name='final_conv_bn')(x)  # 批归一化
    x = Activation('relu', name='final_act')(x)  # 激活函数
    x = GlobalAveragePooling2D(name='final_pooling')(x)  # 全局平均池化
    x = Dense(classes, activation='softmax', name='fc')(x)  # 全连接层

    model = models.Model(img_input, x, name='DenseNet121')  # 创建模型

    return model  # 返回DenseNet模型

def densenet_model(num_classes):
    # 加载预训练的 DenseNet121 模型，不包含顶部的全连接层
    base_model = DenseNet121(weights='/home/huangby/nasnet-densnet-fusion-classification/weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))
    # 冻结预训练模型的所有层，使其在训练过程中不更新权重
    # for layer in base_model.layers:
    #     layer.trainable = False
    # 添加自定义的层
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)
    # 构建新的模型
    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# 主函数
def main():
    model = densenet_model(num_classes=classes)  # 创建DenseNet模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 编译模型
    print(model.summary())  # 打印模型摘要
    modelname = 'densenet'

    # 定义回调函数
    earlystop = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, mode='auto')  # 早停回调
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_model, f"{modelname}.h5"),
        monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)  # 模型检查点回调
    check_path(os.path.join(logs_path, modelname))  # 检查日志路径
    csv_save = CSVLogger(os.path.join(os.path.join(logs_path, modelname), f'{modelname}.csv'))  # CSV日志回调
    reduceLR = ReduceLROnPlateau(monitor='val_accuracy', patience=9, verbose=1, factor=0.5)  # 学习率减半回调

    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # 生成训练、验证和测试数据生成器
    train_generator, validation_generator, test_generator, num_train, num_val, num_test = generate(batch=batch_size,
                                                                                                   shape=(
                                                                                                       width, height),
                                                                                                   ptrain=train_path,
                                                                                                   pval=val_path,
                                                                                                   ptest=test_path)
    start_epoch = 0  # 起始训练轮次
    end_epoch = epochs  # 结束训练轮次
    print("start epoch:", start_epoch, "end epoch:", end_epoch)

    # 训练模型
    history = model.fit_generator(train_generator, validation_data=validation_generator,
                                  verbose=1,
                                  steps_per_epoch=num_train // batch_size,
                                  validation_steps=num_val // batch_size,
                                  epochs=end_epoch,
                                  initial_epoch=start_epoch,
                                  shuffle=True,
                                  callbacks=[checkpoint, csv_save])
    plot_history(history, modelname)  # 绘制训练历史


# 入口函数
if __name__ == '__main__':
    main()  # 执行主函数
