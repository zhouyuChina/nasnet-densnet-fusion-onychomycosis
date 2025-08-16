from flask import Flask, request, jsonify
import base64
import numpy as np
import cv2
import onnxruntime as rt
import os
import threading
from flask import make_response
import json
from collections import Counter

app = Flask(__name__)

# 病害类别列表
dct = ['normal', 'onychomycosis']

# 模型路径配置
MODEL_PATHS = {
    'fusion': './onnx_model/fusion.onnx',
    'densenet': './onnx_model/densenet.onnx',
    'resnet': './onnx_model/resnet.onnx'
}

# 模型缓存和线程锁
model_cache = {}
model_cache_lock = threading.Lock()


def is_blur_fft(image, threshold=-1.4, size=60):
    """
    通过FFT判断图片是否模糊
    :param image: 已解码的灰度图像(numpy数组)
    :param threshold: 高频分量阈值
    :param size: 频域中心区域大小
    :return: True（模糊） / False（清晰）
    """
    try:
        if len(image.shape) == 3:  # 如果是彩色图，转为灰度
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        h, w = gray.shape
        cx, cy = w // 2, h // 2

        # FFT处理
        fft = np.fft.fft2(gray)
        fft_shift = np.fft.fftshift(fft)
        fft_shift[cy - size:cy + size, cx - size:cx + size] = 0  # 屏蔽中心低频区域
        magnitude = np.log(np.abs(fft_shift) + 1e-10)  # 加小值防止log(0)
        mean = np.mean(magnitude)

        app.logger.debug(f"高频能量: {mean:.2f} (阈值: {threshold})")
        return mean < threshold
    except Exception as e:
        app.logger.error(f"模糊检测出错: {str(e)}")
        return False


def load_model(model_name):
    """加载模型并缓存"""
    model_path = MODEL_PATHS[model_name]
    with model_cache_lock:
        if model_path in model_cache:
            return model_cache[model_path]
        else:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            session = rt.InferenceSession(model_path)
            model_cache[model_path] = session
            return session


def model_predict(session, input_data, original_img):
    """
    单个模型的完整预测流程：
    1. 置信度≥85% → 返回具体类别
    2. 置信度<85% → 模糊检测
       - 模糊 → 返回"blurred"
       - 不模糊 → 返回"UNKNOWN"
    """
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: input_data})[0][0]
    top_idx = np.argmax(predictions)
    confidence = float(predictions[top_idx])

    if confidence >= 0.62:
        return dct[top_idx], confidence
    else:
        return "blurred" if is_blur_fft(original_img) else "UNKNOWN", confidence


def preprocess_image(img):
    """图像预处理"""
    processed_img = cv2.resize(img, (224, 224))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    processed_img = processed_img / 255.0
    return np.expand_dims(processed_img, axis=0).astype(np.float32)


@app.route('/predict', methods=['POST'])
def predict():
    # 获取请求参数
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No JSON data received'}), 400

    base64_img = data.get('base64_img')
    if not base64_img:
        return jsonify({'error': 'Missing base64_img parameter'}), 400

    try:
        # Base64解码
        img_data = base64.b64decode(base64_img)
        img_np = np.frombuffer(img_data, dtype=np.uint8)
        img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
        if img is None:
            return jsonify({'error': 'Invalid image data'}), 400

        # 图像预处理
        input_data = preprocess_image(img)

        # 加载所有模型
        models = {
            'fusion': load_model('fusion'),
            'densenet': load_model('densenet'),
            'resnet': load_model('resnet')
        }

        # 各模型独立预测
        model_results = {}
        for name, session in models.items():
            result, confidence = model_predict(session, input_data, img)
            model_results[name] = {
                'result': result,
                'confidence': f"{round(confidence * 100, 2)}%" if isinstance(confidence, float) else "N/A"
            }

        # 结果统计
        all_results = [v['result'] for v in model_results.values()]
        result_counts = Counter(all_results)

        # 最终决策
        if "blurred" in all_results:
            final_result = "blurred"
        elif "UNKNOWN" in all_results:
            final_result = "UNKNOWN"
        else:
            # 只有全部模型都返回具体类别时才进行投票
            final_result = result_counts.most_common(1)[0][0]

        # 准备响应
        response_data = {
            'final_result': final_result,
            'model_results': model_results,
            'result_distribution': dict(result_counts)
        }

        response = make_response(
            json.dumps(response_data, ensure_ascii=False)
        )
        response.headers['Content-Type'] = 'application/json; charset=utf-8'
        return response

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=1008, threaded=True)