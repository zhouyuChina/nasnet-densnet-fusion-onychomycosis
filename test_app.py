# test_app.py（增强版）
import requests
import base64
import os
import json

TEST_IMAGE_PATH = "test_unknown_blurred/test_blurred/images.jpg"
API_URL = "http://localhost:1008/predict"


def test_api():
    # 打印调试信息
    print("当前工作目录:", os.getcwd())
    print("测试图片是否存在:", os.path.exists(TEST_IMAGE_PATH))

    try:
        with open(TEST_IMAGE_PATH, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode()
            f = open(f"test_images/{TEST_IMAGE_PATH.split('/')[-1][:-4]}.txt", 'w')
            f.write(base64_img)
            f.close()

        payload = {
            "base64_img": base64_img,
        }

        # 打印关键参数
        print("BASE64前32位:", base64_img[:32])

        response = requests.post(API_URL, json=payload, timeout=10)

        # 打印完整响应信息
        print("HTTP状态码:", response.status_code)
        print("原始响应:", response.text)

        if response.status_code == 200:
            result = response.json()
            print("预测结果:", json.dumps(result, indent=2))
        else:
            print("错误详情:", response.json())

    except Exception as e:
        print(f"!! 捕获到异常: {str(e)}")


if __name__ == "__main__":
    test_api()
