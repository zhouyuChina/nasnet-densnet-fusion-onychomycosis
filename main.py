from densnet121 import DenseNet
from resnet import resnet_model
from fusion import get_fusion_model
from utils.utils import *  # 导入工具函数
from utils.data_process import generate  # 导入数据生成器函数
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import tensorflow as tf
print("=============", tf.test.is_gpu_available() )

def train_densenet():
    model = DenseNet(reduction=.5, classes=classes)  # 创建DenseNet模型
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

def train_resnet():
    model = resnet_model(num_classes=classes)  # 创建ResNet50模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])  # 编译模型
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

def train_fusion():
    fusion_model = get_fusion_model(num_classes=2)
    modelname = 'fusion'
    earlystop = EarlyStopping(monitor='val_accuracy', patience=15, verbose=1, mode='auto')
    checkpoint = ModelCheckpoint(
        filepath=os.path.join(save_model, f"{modelname}.h5"),
        monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=1)
    check_path(os.path.join(logs_path, modelname))
    csv_save = CSVLogger(os.path.join(os.path.join(logs_path, modelname), f'{modelname}.csv'))
    reduceLR = ReduceLROnPlateau(monitor='val_accuracy', patience=9, verbose=1, factor=0.5)

    # Compile the fusion model
    fusion_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    train_generator, validation_generator, test_generator, num_train, num_val, num_test = generate(batch=batch_size,
                                                                                                   shape=(
                                                                                                       width,
                                                                                                       height),
                                                                                                   ptrain=train_path,
                                                                                                   pval=val_path,
                                                                                                   ptest=test_path)
    start_epoch = 0
    end_epoch = epochs
    print("start epoch:", start_epoch, "end epoch:", end_epoch)
    history = fusion_model.fit_generator(train_generator, validation_data=validation_generator,
                                         verbose=1,
                                         steps_per_epoch=num_train // batch_size,
                                         validation_steps=num_val // batch_size,
                                         epochs=end_epoch,
                                         initial_epoch=start_epoch,
                                         shuffle=True,
                                         callbacks=[checkpoint, csv_save])
    plot_history(history, modelname)


if __name__=='__main__':
    #train_fusion()
    # train_resnet()
    # train_densenet()
    train_fusion()