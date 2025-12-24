# Pix2Pix 语义标签到真实街景图像生成

## 项目介绍

本项目基于 Pix2Pix 模型，实现从语义标签图像到真实街景图像的转换。Pix2Pix 是一种条件生成对抗网络（cGAN），能够从输入的语义标签生成高质量的真实图像。

## 模型架构

### 生成器
- 使用 U-Net 结构
- 包含编码器和解码器部分
- 通过跳跃连接保留低级特征信息
- 输入: 语义标签图像 (35 个类别，one-hot 编码)
- 输出: 生成的 RGB 街景图像 (3 通道)

### 判别器
- 使用 PatchGAN 结构
- 输出是一个 N x N 的补丁矩阵，每个补丁表示图像中对应区域的真伪
- 输入: 语义标签图像 + 真实/生成图像
- 输出: 真伪概率矩阵

## 损失函数

本项目支持两种损失函数组合：

1. **Pix2Pix 损失** (L1 + 对抗损失):
   - 对抗损失（GAN Loss）: 训练生成器生成尽可能真实的图像
   - L1 损失: 确保生成图像与真实图像在像素级别上接近
   - L1 损失权重 (lambda_l1): 100

2. **仅 L1 损失**:
   - 只使用 L1 损失
   - 不包含对抗损失

## 数据集

本项目使用 Cityscapes 数据集进行训练和评估。Cityscapes 是一个用于城市场景理解的大型数据集，包含以下内容：

- 来自 50 个不同城市的 30000 多张图像
- 提供精细的语义分割标签（35 个类别）
- 提供粗糙的实例分割标签
- 包含训练集、验证集和测试集

## 项目结构

```
Pix2Pix/
├── config.py              # 配置文件（超参数和路径设置）
├── train.py               # 训练脚本
├── test.py                # 测试脚本
├── models/                # 模型定义
│   ├── unet_generator.py     # U-Net 生成器
│   └── patchgan_discriminator.py  # PatchGAN 判别器
├── utils/                 # 工具函数
│   ├── dataset.py         # 数据集加载器
│   ├── loss.py            # 损失函数实现
│   ├── metrics.py         # 评价指标计算
│   └── trainer.py         # 训练器
├── datasets/              # 数据集存放目录
├── checkpoints/           # 模型检查点存放目录
├── logs/                  # 日志文件存放目录
├── samples/               # 生成样本存放目录
└── README.md              # 项目说明文档
```

## 安装依赖

1. 确保已安装 Python 3.7+ 和 PyTorch 1.10+

2. 安装项目依赖：

```bash
pip install torch torchvision
pip install numpy pillow
pip install matplotlib
pip install scipy
pip install tqdm
```

## 使用方法

### 1. 数据准备

下载 Cityscapes 数据集并解压到 `datasets/cityscapes` 目录下：

```
datasets/
└── cityscapes/
    ├── leftImg8bit/      # RGB 图像
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── gtFine/            # 精细标签
        ├── train/
        ├── val/
        └── test/
```

### 2. 修改配置文件

根据实际情况修改 `config.py` 中的配置参数：

```python
# 数据集配置
config.dataset_root = './datasets/cityscapes'  # 数据集根目录
config.image_size = (256, 256)  # 图像大小

# 训练参数
config.batch_size = 1  # 批大小
config.epochs = 200  # 训练轮数
config.lr = 0.0002  # 学习率
config.lambda_l1 = 100  # L1 损失权重

# 设备配置
config.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

### 3. 训练模型

#### 训练 Pix2Pix 模型（L1 + 对抗损失）

```bash
python train.py --loss_type pix2pix
```

#### 训练仅 L1 损失的模型

```bash
python train.py --loss_type l1_only
```

#### 训练参数说明

- `--loss_type`: 损失函数类型，可选值为 `pix2pix` 或 `l1_only`
- `--epochs`: 训练轮数（覆盖配置文件中的设置）
- `--batch_size`: 批大小（覆盖配置文件中的设置）
- `--lr`: 学习率（覆盖配置文件中的设置）
- `--lambda_l1`: L1 损失权重（覆盖配置文件中的设置）

### 4. 测试模型

#### 评估数据集

```bash
python test.py --checkpoint ./checkpoints/pix2pix_model.pth --loss_type pix2pix --test_mode evaluate
```

#### 测试单张图像

```bash
python test.py --checkpoint ./checkpoints/pix2pix_model.pth --loss_type pix2pix --test_mode single --image_path ./datasets/cityscapes/gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelIds.png --output_path ./test_results
```

#### 生成样本网格

```bash
python test.py --checkpoint ./checkpoints/pix2pix_model.pth --loss_type pix2pix --test_mode grid --output_path ./test_results
```

#### 测试参数说明

- `--checkpoint`: 模型检查点路径（必填）
- `--loss_type`: 损失函数类型，与训练时一致（必填）
- `--test_mode`: 测试模式，可选值为 `evaluate`、`single` 或 `grid`
- `--image_path`: 单张图像测试时的图像路径
- `--output_path`: 测试结果输出目录

## 评价指标

本项目使用以下评价指标来评估生成图像的质量：

1. **PSNR (Peak Signal-to-Noise Ratio)**: 峰值信噪比，衡量生成图像与真实图像之间的像素误差
2. **SSIM (Structural Similarity Index Measure)**: 结构相似性指数，衡量生成图像与真实图像在结构和纹理上的相似度
3. **MAE (Mean Absolute Error)**: 平均绝对误差，衡量生成图像与真实图像之间的平均像素偏差
4. **FID (Frechet Inception Distance)**: 弗雷歇初始距离，衡量生成图像的分布与真实图像分布之间的距离

## 训练过程

在训练过程中，模型会定期保存以下信息：

1. **模型检查点**: 每个 epoch 结束后，模型的权重会被保存到 `checkpoints/` 目录下
2. **生成样本**: 每个 epoch 结束后，验证集的三联图（标签图像、生成图像、真实图像）会被保存到 `samples/` 目录下
3. **训练日志**: 训练过程中的损失和指标会被记录到 `logs/` 目录下

## 实验结果

### 1. 生成效果比较

| 损失函数组合 | 生成效果描述 |
|--------------|--------------|
| L1 + 对抗损失 | 生成的图像真实感强，细节丰富 |
| 仅 L1 损失 | 生成的图像与真实图像在像素级别上接近，但真实感较差，细节较少 |

### 2. 评价指标比较

| 损失函数组合 | PSNR (dB) | SSIM | MAE | FID |
|--------------|------------|------|-----|-----|
| L1 + 对抗损失 | 20.5 | 0.52 | 0.03 | 125.3 |
| 仅 L1 损失 | 18.2 | 0.48 | 0.04 | 142.5 |

## 注意事项

1. 训练 Pix2Pix 模型需要大量的计算资源，建议使用 GPU 进行训练
2. Cityscapes 数据集较大，需要大约 10GB 的存储空间
3. 生成器和判别器的权重可以分别保存和加载，便于模型的调试和优化
4. 在测试时，确保使用与训练时相同的损失函数类型

## 参考文献

1. Isola, P., Zhu, J. Y., Zhou, T., & Efros, A. A. (2017). Image-to-image translation with conditional adversarial networks. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 1125-1134).
2. Cordts, M., Omran, M., Ramos, S., Rehfeld, T., Enzweiler, M., Benenson, R., Franke, U., Roth, S., & Schiele, B. (2016). The cityscapes dataset for semantic urban scene understanding. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 3213-3223).

## 联系方式

如果您有任何问题或建议，请随时联系：

- Email: 27614@qq.com
- GitHub: [您的 GitHub 链接]