# 导入必要的库和模块
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Sequential
from config import *  # 导入配置文件中的参数
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from utils.utils import *  # 导入工具函数
from utils.data_process import generate  # 导入数据生成器函数
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# 定义第一个残差模块
class Block(layers.Layer):
    def __init__(self, filters, downsample=False, stride=1):
        super(Block, self).__init__()
        self.downsample = downsample  # 是否需要降采样
        self.conv1 = layers.Conv2D(filters, (1, 1), strides=stride, padding='same')  # 1x1卷积
        self.bn1 = layers.BatchNormalization()  # 批归一化
        self.relu = layers.Activation('relu')  # 激活函数
        self.conv2 = layers.Conv2D(filters, (3, 3), strides=1, padding='same')  # 3x3卷积
        self.bn2 = layers.BatchNormalization()  # 批归一化
        self.conv3 = layers.Conv2D(4 * filters, (1, 1), strides=1, padding='same')  # 1x1卷积
        self.bn3 = layers.BatchNormalization()  # 批归一化
        if self.downsample:
            # 如果需要降采样，定义shortcut路径
            self.shortcut = Sequential()
            self.shortcut.add(layers.Conv2D(4 * filters, (1, 1), strides=stride))  # 1x1卷积
            self.shortcut.add(layers.BatchNormalization(axis=3))  # 批归一化

    def call(self, input, training=None):
        # 前向传播
        out = self.conv1(input)  # 1x1卷积
        out = self.bn1(out)  # 批归一化
        out = self.relu(out)  # 激活函数
        out = self.conv2(out)  # 3x3卷积
        out = self.bn2(out)  # 批归一化
        out = self.relu(out)  # 激活函数
        out = self.conv3(out)  # 1x1卷积
        out = self.bn3(out)  # 批归一化
        if self.downsample:
            # 如果需要降采样
            shortcut = self.shortcut(input)  # shortcut路径
        else:
            shortcut = input  # 原始输入
        output = layers.add([out, shortcut])  # 残差连接
        output = tf.nn.relu(output)  # 激活函数
        return output  # 返回输出


# 定义ResNet模型
class ResNet(keras.Model):
    def __init__(self, layer_dims, num_classes=10):
        super(ResNet, self).__init__()
        # 预处理层
        self.padding = keras.layers.ZeroPadding2D((3, 3))  # 零填充
        self.stem = Sequential([
            layers.Conv2D(64, (7, 7), strides=(2, 2)),  # 7x7卷积
            layers.BatchNormalization(),  # 批归一化
            layers.Activation('relu'),  # 激活函数
            layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')  # 最大池化
        ])
        # 定义resblock
        self.layer1 = self.build_resblock(64, layer_dims[0], stride=1)  # 第一个resblock
        self.layer2 = self.build_resblock(128, layer_dims[1], stride=2)  # 第二个resblock
        self.layer3 = self.build_resblock(256, layer_dims[2], stride=2)  # 第三个resblock
        self.layer4 = self.build_resblock(512, layer_dims[3], stride=2)  # 第四个resblock
        # 全局池化层
        self.avgpool = layers.GlobalAveragePooling2D()  # 全局平均池化
        # 全连接层
        self.fc = layers.Dense(num_classes, activation=tf.keras.activations.softmax)  # 全连接层

    def call(self, input, training=None):
        x = self.padding(input)  # 零填充
        x = self.stem(x)  # 通过预处理层
        x = self.layer1(x)  # 通过第一个resblock
        x = self.layer2(x)  # 通过第二个resblock
        x = self.layer3(x)  # 通过第三个resblock
        x = self.layer4(x)  # 通过第四个resblock
        x = self.avgpool(x)  # 全局平均池化
        x = self.fc(x)  # 全连接层
        return x  # 返回输出

    def build_resblock(self, filter_num, blocks, stride=1):
        res_blocks = Sequential()
        if stride != 1 or filter_num * 4 != 64:
            res_blocks.add(Block(filter_num, downsample=True, stride=stride))  # 第一个Block，可能需要降采样
        for pre in range(1, blocks):
            res_blocks.add(Block(filter_num, stride=1))  # 后续的Block
        return res_blocks  # 返回resblock


# # 定义ResNet50模型
# def ResNet50(num_classes=10):
#     return ResNet([3, 4, 6, 3], num_classes=num_classes)  # ResNet-50的层数配置

def resnet_model(num_classes):
    # 加载预训练的 ResNet50 模型，不包含顶部的全连接层
    base_model = ResNet50(weights='/home/huangby/nasnet-densnet-fusion-classification/weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(224, 224, 3))
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
if __name__ == '__main__':
    model = resnet_model(num_classes=classes)  # 创建ResNet50模型
    modelname = 'resnet'  # 模型名称
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
