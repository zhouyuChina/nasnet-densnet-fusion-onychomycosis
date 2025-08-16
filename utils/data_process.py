"""
数据的特征工程
图像的归一化、随机缩放、随机旋转、宽度仿射，高度仿射，随机裁剪，需要注意的是，只需要对训练进行特征工程
"""

import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def prep_fn(img):
    # img = img.astype(np.float32) / 255.0
    # img = (img - 0.5) * 2
    img = img/255.
    return img

# gen = ImageDataGenerator(preprocessing_function=prep_fn
# )

def generate(batch, shape, ptrain, pval, ptest):
    datagen1 = ImageDataGenerator(preprocessing_function=prep_fn,shear_range=0.2,zoom_range=0.2,rotation_range=90,width_shift_range=0.2,height_shift_range=0.2,horizontal_flip=True)
    #datagen1 = ImageDataGenerator(preprocessing_function=prep_fn)
    datagen2 = ImageDataGenerator(preprocessing_function=prep_fn)
    datagen3 = ImageDataGenerator(preprocessing_function=prep_fn)

    train_generator = datagen1.flow_from_directory(
        directory=ptrain,
        target_size=shape,
        batch_size=batch,
        class_mode='categorical',
    shuffle=True)
    print(train_generator.class_indices)

    validation_generator = datagen2.flow_from_directory(
        directory=pval,
        target_size=shape,
        batch_size=batch,
        class_mode='categorical',
    shuffle=True)

    test_generator = datagen3.flow_from_directory(
        directory=ptest,
        target_size=shape,
        batch_size=batch,
        class_mode='categorical',
        shuffle=True)

    num_train, num_val, num_test = 0, 0, 0
    for f in os.listdir(ptrain):
        num_train += len(os.listdir(os.path.join(ptrain, f)))  #

    for v in os.listdir(pval):
        num_val += len(os.listdir(os.path.join(pval, v)))
        # num_val += len(os.listdir(os.path.join(ptrain, v)))
    for v in os.listdir(ptest):
        num_test += len(os.listdir(os.path.join(ptest, v)))

    return train_generator, validation_generator, test_generator, num_train, num_val, num_test







