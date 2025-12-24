# 测试脚本
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from config import config
from models.unet_generator import UNetGenerator
from models.patchgan_discriminator import PatchGANDiscriminator
from utils.dataset import CityscapesDataset, CityscapesTransform
from utils.loss import Pix2PixLoss, L1OnlyLoss
from utils.metrics import MetricsCalculator
from utils.trainer import Trainer

class Tester:
    """Pix2Pix模型测试器"""
    def __init__(self, config, checkpoint_path, loss_type='pix2pix'):
        """初始化
        参数：
            config: 配置对象
            checkpoint_path: 模型检查点路径
            loss_type: 损失函数类型
        """
        self.config = config
        self.device = config.device
        self.checkpoint_path = checkpoint_path
        self.loss_type = loss_type
        
        # 创建模型
        self.generator = UNetGenerator(
            in_channels=config.gen_in_channels,
            out_channels=config.gen_out_channels,
            features=config.gen_features
        ).to(self.device)
        
        self.discriminator = PatchGANDiscriminator(
            in_channels=config.dis_in_channels,
            features=config.dis_features
        ).to(self.device)
        
        # 创建损失函数
        if loss_type == 'pix2pix':
            self.loss_fn = Pix2PixLoss(lambda_l1=config.lambda_l1)
        elif loss_type == 'l1_only':
            self.loss_fn = L1OnlyLoss()
        else:
            raise ValueError(f"未知的损失函数类型: {loss_type}")
        
        # 创建评价指标计算器
        self.metrics_calculator = MetricsCalculator(device=config.device)
        
        # 加载模型权重
        self._load_checkpoint()
        
    def _load_checkpoint(self):
        """加载模型检查点"""
        print(f"加载模型检查点: {self.checkpoint_path}")
        
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"检查点文件不存在: {self.checkpoint_path}")
        
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 加载模型权重
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        
        print(f"模型检查点加载完成 (epoch: {checkpoint.get('epoch', 'unknown')})")
        
    def test_single_image(self, label_path):
        """测试单张图像
        参数：
            label_path: 语义标签图像路径
        返回：
            生成的真实图像
        """
        self.generator.eval()
        
        # 加载并预处理标签图像
        label = Image.open(label_path)
        
        transform = CityscapesTransform(image_size=config.image_size)
        
        # 由于transform需要同时处理图像和标签，这里创建一个虚拟图像
        dummy_image = Image.new('RGB', label.size)
        _, processed_label = transform(dummy_image, label)
        
        # 添加批量维度
        processed_label = processed_label.unsqueeze(0).to(self.device)
        
        # 生成图像
        with torch.no_grad():
            generated_image = self.generator(processed_label)
        
        # 移除批量维度并转换为PIL图像
        generated_image = generated_image.squeeze(0).detach().cpu()
        generated_image = (generated_image + 1) / 2  # 反归一化
        generated_image = generated_image.permute(1, 2, 0).numpy() * 255
        generated_image = Image.fromarray(generated_image.astype(np.uint8))
        
        return generated_image
        
    def evaluate_dataset(self, dataset, batch_size=1):
        """评估整个数据集
        参数：
            dataset: 测试数据集
            batch_size: 批大小
        返回：
            平均损失和指标
        """
        self.generator.eval()
        self.discriminator.eval()
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        
        total_losses = {}
        total_metrics = {}
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                print(f"评估进度: {i + 1}/{len(dataloader)}")
                
                real_images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 生成图像
                generated_images = self.generator(labels)
                
                # 计算损失
                losses = self.loss_fn(self.discriminator, real_images, generated_images, labels)
                
                # 记录损失
                for key, value in losses.items():
                    if key not in total_losses:
                        total_losses[key] = []
                    total_losses[key].append(value.item())
                
                # 计算和记录指标
                metrics = self.metrics_calculator.calculate_all_metrics(generated_images, real_images)
                for key, value in metrics.items():
                    if key not in total_metrics:
                        total_metrics[key] = []
                    if value is not None:
                        total_metrics[key].append(value)
        
        # 计算平均损失和指标
        avg_losses = {key: np.mean(value) for key, value in total_losses.items()}
        avg_metrics = {key: np.mean(value) for key, value in total_metrics.items()}
        
        return avg_losses, avg_metrics
        
    def generate_sample_grid(self, dataset, num_samples=4, save_path=None):
        """生成样本网格
        参数：
            dataset: 数据集
            num_samples: 样本数量
            save_path: 保存路径
        """
        self.generator.eval()
        
        # 随机选择样本
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
        # 创建图像网格
        grid_images = []
        
        for i, idx in enumerate(indices):
            print(f"生成样本 {i + 1}/{num_samples}")
            
            batch = dataset[idx]
            real_image = batch['image'].unsqueeze(0).to(self.device)
            label = batch['label'].unsqueeze(0).to(self.device)
            
            # 生成图像
            with torch.no_grad():
                generated_image = self.generator(label)
            
            # 转换标签为可视化格式
            label_visual = torch.argmax(label[0], dim=0, keepdim=True)
            label_visual = label_visual.float() / label_visual.max()
            label_visual = label_visual.expand(3, -1, -1)
            
            # 添加到网格
            grid_images.append(label_visual)
            grid_images.append(generated_image[0])
            grid_images.append(real_image[0])
        
        # 拼接成网格
        grid = torch.cat(grid_images, dim=2)
        
        # 保存图像
        if save_path:
            save_image(grid, save_path, normalize=True)
            print(f"样本网格已保存至: {save_path}")
        
        return grid

if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Pix2Pix模型测试')
    parser.add_argument('--checkpoint', type=str, required=True, 
                        help='模型检查点路径')
    parser.add_argument('--loss_type', type=str, default='l1_only',
                        help='损失函数类型: pix2pix (L1+GAN) 或 l1_only (仅L1)')
    parser.add_argument('--test_mode', type=str, default='evaluate', 
                        help='测试模式: evaluate (评估数据集) 或 single (测试单张图像) 或 grid (生成样本网格)')
    parser.add_argument('--image_path', type=str, default=None, 
                        help='单张图像测试时的图像路径')
    parser.add_argument('--output_path', type=str, default='./test_results', 
                        help='测试结果输出目录')
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    print("=" * 50)
    print("Pix2Pix模型测试")
    print("=" * 50)
    
    # 创建测试器
    tester = Tester(config, args.checkpoint, args.loss_type)
    
    if args.test_mode == 'evaluate':
        print("\n评估模式")
        print("加载测试数据集...")
        
        # 创建测试数据集
        transform = CityscapesTransform(image_size=config.image_size)
        test_dataset = CityscapesDataset(
            root_dir=config.dataset_root,
            split='val',  # 通常使用验证集作为测试集
            transform=transform
        )
        
        print(f"测试集大小: {len(test_dataset)}")
        
        # 评估数据集
        avg_losses, avg_metrics = tester.evaluate_dataset(test_dataset, batch_size=config.test_batch_size)
        
        # 打印评估结果
        print("\n评估结果:")
        print("平均损失:")
        for key, value in avg_losses.items():
            print(f"  {key}: {value:.4f}")
        
        print("\n平均指标:")
        for key, value in avg_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # 保存评估结果到文件
        result_file = os.path.join(args.output_path, 'evaluation_result.txt')
        with open(result_file, 'w') as f:
            f.write("Pix2Pix模型评估结果\n")
            f.write(f"评估时间: {torch.tensor(0).float().item()}\n")  # 这里可以添加实际时间
            f.write(f"测试集大小: {len(test_dataset)}\n\n")
            
            f.write("平均损失:\n")
            for key, value in avg_losses.items():
                f.write(f"  {key}: {value:.4f}\n")
            
            f.write("\n平均指标:\n")
            for key, value in avg_metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
        
        print(f"\n评估结果已保存至: {result_file}")
        
    elif args.test_mode == 'single':
        print("\n单张图像测试模式")
        
        if not args.image_path:
            raise ValueError("单张图像测试时必须提供 --image_path 参数")
        
        # 测试单张图像
        generated_image = tester.test_single_image(args.image_path)
        
        # 保存生成的图像
        output_image_path = os.path.join(args.output_path, 'generated_image.png')
        generated_image.save(output_image_path)
        
        print(f"生成图像已保存至: {output_image_path}")
        
    elif args.test_mode == 'grid':
        print("\n样本网格生成模式")
        
        # 创建测试数据集
        transform = CityscapesTransform(image_size=config.image_size)
        test_dataset = CityscapesDataset(
            root_dir=config.dataset_root,
            split='val',
            transform=transform
        )
        
        # 生成样本网格
        grid_save_path = os.path.join(args.output_path, 'sample_grid.png')
        tester.generate_sample_grid(test_dataset, num_samples=4, save_path=grid_save_path)
        
    else:
        raise ValueError(f"未知的测试模式: {args.test_mode}")
    
    print("\n" + "=" * 50)
    print("测试完成")
    print("=" * 50)