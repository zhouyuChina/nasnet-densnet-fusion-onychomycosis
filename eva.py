# test_app.py（增强版）
import requests
import base64
import os
import json

API_URL = "http://localhost:1008/predict"


def test_api():
    # 打印调试信息
    try:
        with open(TEST_IMAGE_PATH, "rb") as f:
            base64_img = base64.b64encode(f.read()).decode()
            f = open(f"test_images/{TEST_IMAGE_PATH.split('/')[-1][:-4]}.txt", 'w')
            f.write(base64_img)
            f.close()

        payload = {
            "base64_img": base64_img,
        }

        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()
            print("预测结果:", result)
            return result['final_result']
        else:
            print("错误详情:", response.json())
            return None

    except Exception as e:
        print(f"!! 捕获到异常: {str(e)}")


if __name__ == "__main__":
    trues = []
    preds = []
    paths = 'dataset'
    for file in os.listdir(paths):
        for image in os.listdir(paths + "/" + file):
            TEST_IMAGE_PATH = paths + "/" + file + "/" + image
            r = test_api()
            preds.append(r)
            trues.append(file)
