import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from config import *
from utils.data_process import generate

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
from densnet121 import DenseNet
from utils.utils import *
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Conv2D, Flatten, Dense, Input
from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, Concatenate
from tensorflow.keras.models import Model

# 获取所有GPU组成list
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 0:
    # 对需要进行限制的GPU进行设置
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=1024 * 20)])


def get_fusion_model(num_classes):
    # 输入
    input_tensor = Input(shape=(224, 224, 3))
    # 加载预训练的DenseNet121和ResNet50模型，不包括顶层分类器
    densenet_model = DenseNet121(weights='weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                 include_top=False, input_tensor=input_tensor)
    resnet_model = ResNet50(weights='weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False,input_tensor=input_tensor)

    # 手动为DenseNet121的层添加前缀
    for layer in densenet_model.layers:
        layer._name = 'densenet_' + layer.name
    # 手动为ResNet50的层添加前缀
    for layer in resnet_model.layers:
        layer._name = 'resnet_' + layer.name
    # 提取两个模型的特征
    densenet_features = densenet_model.output
    resnet_features = resnet_model.output
    # 对特征进行全局平均池化
    densenet_features = GlobalAveragePooling2D()(densenet_features)
    resnet_features = GlobalAveragePooling2D()(resnet_features)
    # 融合特征
    merged_features = Concatenate()([densenet_features, resnet_features])
    # 添加自定义的全连接层进行分类
    x = Dense(512, activation='relu')(merged_features)
    output = Dense(num_classes, activation='softmax')(x)
    # 构建融合模型
    model = Model(inputs=input_tensor, outputs=output)
    # 编译模型
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

if __name__ == '__main__':
    fusion_model = get_fusion_model(num_classes=classes)
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
