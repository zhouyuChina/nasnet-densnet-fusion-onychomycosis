# 甲真菌病检测系统

基于深度学习的甲真菌病（灰指甲）自动检测系统，使用DenseNet和ResNet融合模型进行图像分类。

## 🎯 项目概述

本项目实现了一个完整的甲真菌病检测系统，包括：
- 深度学习模型训练
- 模型融合优化
- Web API服务
- 图像质量检测
- 实时推理服务

## 🏗️ 技术架构

### 核心模型
- **DenseNet121**: 密集连接网络
- **ResNet50**: 残差网络
- **融合模型**: DenseNet + ResNet特征融合

### 技术栈
- **深度学习**: TensorFlow 2.15.0
- **推理引擎**: ONNX Runtime
- **Web框架**: Flask
- **图像处理**: OpenCV
- **数据处理**: NumPy, Pandas, Scikit-learn

## 📁 项目结构

```
├── 模型训练
│   ├── main.py              # 主训练脚本
│   ├── densnet121.py        # DenseNet模型
│   ├── resnet.py            # ResNet模型
│   ├── fusion.py            # 模型融合
│   ├── config.py            # 配置文件
│   └── evaluate.py          # 模型评估
│
├── Web服务
│   ├── app.py               # Flask API服务
│   ├── test_app.py          # API测试
│   └── blurred_detect.py    # 模糊检测
│
├── 数据处理
│   ├── split_data.py        # 数据集分割
│   ├── utils/               # 工具函数
│   └── dataset/             # 原始数据集
│
├── 模型文件
│   ├── onnx_model/          # ONNX格式模型
│   ├── weights/             # 预训练权重
│   └── save/                # 训练保存模型
│
└── 文档
    ├── requirements.txt     # 依赖包
    └── 接口说明.docx        # API文档
```

## 🚀 快速开始

### 环境要求
- Python 3.11.5
- TensorFlow 2.15.0
- CUDA支持（可选，用于GPU加速）

### 安装依赖
```bash
pip install -r requirements.txt
```

### 模型训练
```bash
# 训练DenseNet模型
python main.py --model densenet

# 训练ResNet模型
python main.py --model resnet

# 训练融合模型
python main.py --model fusion
```

### 启动Web服务
```bash
python app.py
```

### API测试
```bash
python test_app.py
```

## 📊 数据集

- **类别**: 2类（正常/甲真菌病）
- **数据分割**: 训练70% / 验证10% / 测试20%
- **图像尺寸**: 224×224像素

## 🔧 配置说明

主要配置参数在 `config.py` 中：

```python
width = 224          # 图像宽度
height = 224         # 图像高度
batch_size = 4       # 批次大小
learning_rate = 1e-4 # 学习率
epochs = 500         # 训练轮数
classes = 2          # 类别数
```

## 🌐 API接口

### 预测接口
- **URL**: `/predict`
- **方法**: POST
- **输入**: Base64编码的图像
- **输出**: JSON格式的预测结果

### 响应格式
```json
{
    "status": "success",
    "prediction": "onychomycosis",
    "confidence": 0.85,
    "model_used": "fusion"
}
```

## 🎯 核心功能

### 1. 图像分类
- 正常指甲 vs 甲真菌病二分类
- 置信度阈值控制（≥62%）
- 多模型投票机制

### 2. 图像质量检测
- FFT模糊检测算法
- 自动过滤低质量图像
- 返回"blurred"状态

### 3. 模型融合
- DenseNet + ResNet特征融合
- 提高分类精度
- 增强模型鲁棒性

### 4. 实时推理
- ONNX模型优化
- 模型缓存机制
- 线程安全设计

## 📈 性能指标

- **准确率**: >90%
- **推理速度**: <100ms
- **支持并发**: 多线程处理

## 🔒 安全特性

- 输入验证和清理
- 错误处理和日志记录
- 模型缓存优化
- 线程安全设计

## 📝 使用说明

1. **准备数据**: 将图像放入 `dataset/` 目录
2. **训练模型**: 运行训练脚本
3. **启动服务**: 运行Flask应用
4. **调用API**: 发送图像进行预测

## 🤝 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证。

## 📞 联系方式

如有问题或建议，请提交 Issue 或联系项目维护者。

---

**注意**: 本项目仅用于研究和学习目的，不构成医疗建议。实际医疗诊断请咨询专业医生。
