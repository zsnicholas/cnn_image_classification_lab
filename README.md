# CNN Image Classification Project

一个基于 PyTorch 的图像分类项目，实现了 MNIST 手写数字识别和 CIFAR-10 彩色图像分类，覆盖数据加载、模型构建、训练评估、模型保存、模型加载以及本地图片预测等完整流程。

项目使用自定义 CNN 网络完成图像分类任务，并提供训练脚本、预测脚本和模型权重文件，方便复现训练结果和进行单张图片推理。

## 项目特点

- 基于 PyTorch 搭建 CNN 图像分类模型
- 支持 MNIST 手写数字识别
- 支持 CIFAR-10 彩色图像分类
- 支持 CIFAR-10 数据增强训练
- 支持训练过程中自动保存最优模型
- 支持加载已训练模型进行预测
- 支持测试集随机样本预测
- 支持本地图片输入预测
- 代码结构清晰，按模型、工具函数、训练脚本和预测脚本进行模块化组织

## 技术栈

- Python
- PyTorch
- Torchvision
- Pillow
- CNN
- DataLoader
- CrossEntropyLoss
- Adam Optimizer

## 项目结构

```text
cnn_image_classification_lab/
├── checkpoints/
│   ├── best_mnist_cnn.pth
│   ├── best_cifar10_cnn.pth
│   └── best_cifar10_aug_cnn.pth
├── data/
│   ├── MNIST/
│   └── cifar-10-batches-py/
├── models/
│   ├── simple_cnn.py
│   └── cifar_cnn.py
├── test_images/
│   ├── mnist_test.png
│   └── cifar_test.jpg
├── utils/
│   ├── checkpoint_utils.py
│   └── train_utils.py
├── train_mnist.py
├── train_cifar10.py
├── predict_mnist.py
├── predict_cifar10.py
├── predict_local_mnist.py
├── predict_local_cifar10.py
└── README.md
```

说明：

- `data/` 用于保存运行脚本时自动下载的数据集。
- `checkpoints/` 用于保存训练过程中生成的模型权重文件。
- `test_images/` 用于保存本地预测时手动准备的测试图片。

## 功能说明

### MNIST 手写数字识别

项目使用 `SimpleCNN` 对 MNIST 手写数字图片进行分类，输入为 28x28 单通道灰度图，输出为 0 到 9 的数字类别。

模型文件：

```text
models/simple_cnn.py
```

训练脚本：

```text
train_mnist.py
```

预测脚本：

```text
predict_mnist.py
predict_local_mnist.py
```

### CIFAR-10 彩色图像分类

项目使用 `CIFAR10CNN` 对 CIFAR-10 彩色图片进行分类，输入为 32x32 三通道 RGB 图像，输出为 10 个类别之一。

CIFAR-10 类别包括：

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

模型文件：

```text
models/cifar_cnn.py
```

训练脚本：

```text
train_cifar10.py
```

预测脚本：

```text
predict_cifar10.py
predict_local_cifar10.py
```

## 安装依赖

建议使用 Python 3.9 或以上版本。

```bash
pip install torch torchvision pillow
```

如果需要使用 GPU，请根据本机 CUDA 版本安装对应的 PyTorch 版本。

## 数据集来源

项目使用 Torchvision 内置数据集接口加载 MNIST 和 CIFAR-10。

第一次运行训练或测试脚本时，如果本地没有对应数据集，程序会自动下载到 `data/` 目录：

```python
datasets.MNIST(root="./data", download=True)
datasets.CIFAR10(root="./data", download=True)
```


## 训练模型

### 训练 MNIST 模型

```bash
python train_mnist.py
```

训练过程中会自动下载 MNIST 数据集，并保存测试集准确率最高的模型：

```text
checkpoints/best_mnist_cnn.pth
```

默认配置：

```text
batch size: 64
epochs: 3
optimizer: Adam
learning rate: 0.001
loss function: CrossEntropyLoss
```

### 训练 CIFAR-10 模型

```bash
python train_cifar10.py
```

训练过程中会自动下载 CIFAR-10 数据集，并使用数据增强提升模型泛化能力。

使用的数据增强方式：

```text
RandomCrop
RandomHorizontalFlip
Normalize
```

模型默认保存路径：

```text
checkpoints/best_cifar10_aug_cnn.pth
```

默认配置：

```text
batch size: 64
epochs: 10
optimizer: Adam
learning rate: 0.001
loss function: CrossEntropyLoss
```

## 模型预测

### 使用 MNIST 测试集预测

```bash
python predict_mnist.py
```

脚本会从 MNIST 测试集中随机选取一张图片，并输出：

```text
真实标签
预测标签
预测概率
各类别概率
```

### 使用 CIFAR-10 测试集预测

```bash
python predict_cifar10.py
```

脚本会从 CIFAR-10 测试集中随机选取一张图片，并输出：

```text
真实类别
预测类别
预测概率
各类别概率
```

## 本地图片预测

本地图片需要自己准备并放到 `test_images/` 目录下。

可以按下面的方式准备目录：

```bash
mkdir test_images
```

### 预测本地 MNIST 图片

默认图片路径：

```text
test_images/mnist_test.png
```

运行：

```bash
python predict_local_mnist.py
```

脚本会将本地图片处理为模型需要的 28x28 灰度图格式。

建议使用黑底或白底的单个手写数字图片进行测试，并将文件命名为：

```text
test_images/mnist_test.png
```

### 预测本地 CIFAR-10 图片

默认图片路径：

```text
test_images/cifar_test.jpg
```

运行：

```bash
python predict_local_cifar10.py
```

脚本会将本地图片处理为模型需要的 32x32 RGB 图像格式。

建议使用属于 CIFAR-10 类别范围内的图片进行测试，例如飞机、汽车、鸟、猫、狗、船、卡车等，并将文件命名为：

```text
test_images/cifar_test.jpg
```

## 核心模块

### `models/simple_cnn.py`

定义 MNIST 分类使用的 CNN 模型。

网络结构：

```text
Conv2d -> ReLU -> MaxPool2d
Conv2d -> ReLU -> MaxPool2d
Flatten -> Linear
```

### `models/cifar_cnn.py`

定义 CIFAR-10 分类使用的 CNN 模型。

网络结构：

```text
Conv2d -> ReLU -> MaxPool2d
Conv2d -> ReLU -> MaxPool2d
Conv2d -> ReLU -> MaxPool2d
Flatten -> Linear -> ReLU -> Linear
```

### `utils/train_utils.py`

封装训练和评估流程：

```text
train_one_epoch
evaluate
```

主要负责：

```text
模型训练
损失计算
准确率统计
测试集评估
```

### `utils/checkpoint_utils.py`

封装模型权重保存和加载逻辑：

```text
save_checkpoint
load_checkpoint
```

主要负责：

```text
保存最优模型
加载已训练模型
恢复模型参数
```

## 说明

- 项目会根据当前环境自动选择 CPU 或 CUDA。
- 第一次运行训练脚本时会自动下载对应数据集。
- 本地图片预测需要提前准备好模型权重文件和测试图片。
- `train_cifar10.py` 默认保存增强训练后的模型：`best_cifar10_aug_cnn.pth`。
- `predict_cifar10.py` 和 `predict_local_cifar10.py` 当前默认加载：`best_cifar10_cnn.pth`。
- 如果希望使用增强训练后的 CIFAR-10 模型，可以将预测脚本中的模型路径修改为：

```text
checkpoints/best_cifar10_aug_cnn.pth
```

## 可扩展方向

- 增加 BatchNorm 和 Dropout
- 引入 ResNet 等更深层 CNN 结构
- 增加训练日志可视化
- 增加混淆矩阵和分类报告
- 支持命令行参数配置训练轮数、学习率和模型路径
- 支持更多图像分类数据集
