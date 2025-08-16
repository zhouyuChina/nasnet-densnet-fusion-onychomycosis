# %%
import numpy as np
import onnxruntime as rt
import cv2

dct = ['normal', 'onychomycosis']

onnx_name = './onnx_model/resnet.onnx'
onnx_model = rt.InferenceSession(onnx_name)
input_name = onnx_model.get_inputs()[0].name
test_image = f'dataset/onychomycosis/12.jpg'  # 输入你要识别的图片路径
img = cv2.imdecode(np.fromfile(file=test_image, dtype=np.uint8), cv2.IMREAD_COLOR)
x = cv2.resize(img, (224, 224))
x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
x = x / 255
x = np.expand_dims(x, axis=0)
# 对置信度进行排序
predictions = onnx_model.run(None, {input_name: x.astype(np.float32)})[0][0]
print(predictions)
sorted_indices = np.argsort(predictions)[::-1]
# 输出每个类别的置信度
print("预测结果（按置信度从高到低排序）:")
for index in sorted_indices:
    print(f"类别 {dct[index]}: 置信度 {predictions[index] * 100:.2f}%")
