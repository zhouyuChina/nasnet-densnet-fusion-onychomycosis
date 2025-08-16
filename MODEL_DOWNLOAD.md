# 模型文件下载说明

由于GitHub对文件大小有限制（单个文件不能超过100MB），模型文件无法直接存储在Git仓库中。

## 📁 需要的模型文件

### ONNX模型文件
- `onnx_model/densenet.onnx` (27MB)
- `onnx_model/resnet.onnx` (90MB) 
- `onnx_model/fusion.onnx` (122MB)

### 预训练权重文件
- `weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5` (28MB)
- `weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5` (90MB)

## 🔧 获取模型文件的方法

### 方法1: 训练生成模型
```bash
# 1. 准备数据集
python split_data.py

# 2. 训练模型
python main.py --model densenet
python main.py --model resnet  
python main.py --model fusion

# 3. 转换为ONNX格式
python convert2onnx.py
```

### 方法2: 下载预训练权重
```bash
# 创建weights目录
mkdir -p weights

# 下载DenseNet121预训练权重
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.8/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5 -O weights/densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5

# 下载ResNet50预训练权重
wget https://github.com/fchollet/deep-learning-models/releases/download/v0.6/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5 -O weights/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
```

### 方法3: 从其他存储服务下载
如果上述链接不可用，可以从以下地址下载：

- **百度网盘**: [链接待补充]
- **阿里云OSS**: [链接待补充]
- **Google Drive**: [链接待补充]

## 📋 文件结构检查

确保您的项目结构如下：
```
project/
├── onnx_model/
│   ├── densenet.onnx
│   ├── resnet.onnx
│   └── fusion.onnx
├── weights/
│   ├── densenet121_weights_tf_dim_ordering_tf_kernels_notop.h5
│   └── resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
└── ...其他文件
```

## ⚠️ 注意事项

1. **文件大小**: 确保下载的文件大小正确
2. **文件完整性**: 下载完成后验证文件完整性
3. **路径正确**: 确保文件放在正确的目录中
4. **权限设置**: 确保文件有读取权限

## 🚀 验证安装

下载完成后，运行以下命令验证：

```bash
# 检查文件是否存在
ls -la onnx_model/
ls -la weights/

# 测试模型加载
python test_app.py
```

## 📞 获取帮助

如果遇到问题，请：
1. 检查文件路径是否正确
2. 确认文件大小是否正常
3. 查看错误日志
4. 提交Issue获取帮助
