# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from tensorflow.keras.models import load_model
import tf2onnx
import onnxruntime as rt
import numpy as np
from densnet121 import DenseNet
from sklearn.metrics import accuracy_score
from utils.utils import *
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from utils.data_process import prep_fn
from tqdm import tqdm
from config import *
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import NASNetMobile
from tensorflow.keras import optimizers
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, Reshape, Dropout
from fusion import get_fusion_model
from densnet121 import densenet_model
from resnet import resnet_model
from config import *
def relu6(x):
    return K.relu(x, max_value=6)


from tqdm import tqdm
# import cv2
#%%
onnx_name = './onnx_model/resnet.onnx'
#%%
# model = get_fusion_model(num_classes=classes)
# model.load_weights("save/fusion.h5")
model = resnet_model(num_classes=classes)
#resnet.build(input_shape=(None, width, height, 3))
model.load_weights("save/resnet.h5")
# model = load_model("./save/nasnet.h5")
tf2onnx.convert.from_keras(model, output_path=onnx_name)


# %%
onnx_model = rt.InferenceSession(onnx_name)
input_name = onnx_model.get_inputs()[0].name
result = onnx_model.run(None, {input_name: np.random.randn(1, 224, 224, 3).astype(np.float32)})
# %%
# from config import test_path, class_dict

# dct = class_dict
# trues, predictions = [], []
# width, height = 224,224
# for classes in os.listdir(test_path):
#     for img in os.listdir(os.path.join(test_path, classes)):
#         # image = Image.open(os.path.join(os.path.join(test_path, classes), img))
#         img = cv2.imread(os.path.join(os.path.join(test_path, classes), img))
#         x = cv2.resize(img, (224,224))
#         x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
#         x = x / 255
#         x = np.expand_dims(x, axis=0)
#         predict = np.argmax(onnx_model.run(None, {input_name: x.astype(np.float32)})[0], 1)[0]
#         predict_label = dct[predict]
#         trues.append(classes)
#         predictions.append(predict_label)
#         print("predict===", predict_label.lower(),'True====', classes.lower())

