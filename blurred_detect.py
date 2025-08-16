import cv2
import numpy as np
import os
import shutil


def is_blur(image_path, threshold=100):
    """支持中文路径的模糊检测"""
    try:
        with open(image_path, 'rb') as f:
            img_data = np.frombuffer(f.read(), dtype=np.uint8)
        image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)

        if image is None:
            return False

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        score = cv2.Laplacian(gray, cv2.CV_64F).var()
        return score < threshold
    except:
        return False


import cv2
import numpy as np


def is_blur_fft(image_path, threshold=10, size=60):
    """
    通过FFT判断图片是否模糊（支持中文路径）
    :param image_path: 图片路径（支持中文）
    :param threshold: 高频分量阈值（默认10）
    :param size: 频域中心区域大小（默认60）
    :return: True（模糊） / False（清晰）
    """
    try:
        # 支持中文路径的读取方式
        with open(image_path, 'rb') as f:
            img_data = np.frombuffer(f.read(), dtype=np.uint8)
            image = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)  # 关键点：使用imdecode

        if image is None:
            print(f"警告：图片解码失败 {image_path}")
            return False

        h, w = image.shape
        cx, cy = w // 2, h // 2

        # FFT处理
        fft = np.fft.fft2(image)
        fft_shift = np.fft.fftshift(fft)
        fft_shift[cy - size:cy + size, cx - size:cx + size] = 0  # 屏蔽中心低频区域
        magnitude = np.log(np.abs(fft_shift) + 1e-10)  # 加小值防止log(0)
        mean = np.mean(magnitude)

        print(f"{os.path.basename(image_path)} - 高频能量: {mean:.2f} (阈值: {threshold})")
        return mean < threshold

    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {str(e)}")
        return False

print(is_blur_fft(f"test_blurred/images.jpg", threshold=-1.4))
