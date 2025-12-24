# 主训练脚本
import sys
import os

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config
from models.unet_generator import UNetGenerator
from models.patchgan_discriminator import PatchGANDiscriminator
from utils.dataset import CityscapesDataset, CityscapesTransform
from utils.loss import Pix2PixLoss, L1OnlyLoss
from utils.metrics import MetricsCalculator
from utils.trainer import Trainer
from torch.utils.data import DataLoader

if __name__ == '__main__':
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Pix2Pix模型训练')
    parser.add_argument('--loss_type', type=str, default='l1_only',
                        help='损失函数类型: pix2pix (L1+GAN) 或 l1_only (仅L1)')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='加载模型检查点的路径')
    parser.add_argument('--start_epoch', type=int, default=0, 
                        help='开始训练的轮次')
    args = parser.parse_args()
    
    print("=" * 50)
    print("Pix2Pix模型训练")
    print("=" * 50)
    
    # 打印配置信息
    print("配置信息:")
    print(f"  设备: {config.device}")
    print(f"  数据集根目录: {config.dataset_root}")
    print(f"  图像大小: {config.image_size}")
    print(f"  批大小: {config.batch_size}")
    print(f"  训练轮次: {config.num_epochs}")
    print(f"  学习率: {config.lr}")
    print(f"  损失函数权重lambda_l1: {config.lambda_l1}")
    print(f"  损失函数类型: {args.loss_type}")
    print(f"  检查点: {args.checkpoint}")
    print("=" * 50)
    
    # 创建数据变换
    transform = CityscapesTransform(image_size=config.image_size)
    
    # 创建训练数据集
    print("加载训练数据集...")
    train_dataset = CityscapesDataset(
        root_dir=config.dataset_root,
        split='train',
        transform=transform
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0  # Windows系统下建议设为0
    )
    
    # 创建验证数据集
    print("加载验证数据集...")
    val_dataset = CityscapesDataset(
        root_dir=config.dataset_root,
        split='val',
        transform=transform
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.test_batch_size,
        shuffle=False,
        num_workers=0  # Windows系统下建议设为0
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print("=" * 50)
    
    # 创建模型
    print("创建模型...")
    generator = UNetGenerator(
        in_channels=config.gen_in_channels,
        out_channels=config.gen_out_channels,
        features=config.gen_features
    )
    
    discriminator = PatchGANDiscriminator(
        in_channels=config.dis_in_channels,
        features=config.dis_features
    )
    
    # 创建损失函数
    print(f"创建损失函数 ({args.loss_type})...")
    if args.loss_type == 'pix2pix':
        loss_fn = Pix2PixLoss(lambda_l1=config.lambda_l1)
    elif args.loss_type == 'l1_only':
        loss_fn = L1OnlyLoss()
    else:
        raise ValueError(f"未知的损失函数类型: {args.loss_type}")
    
    # 创建评价指标计算器
    metrics_calculator = MetricsCalculator(device=config.device)
    
    # 创建训练器
    print("创建训练器...")
    trainer = Trainer(config, generator, discriminator, loss_fn, metrics_calculator)
    
    # 加载模型检查点（如果提供）
    if args.checkpoint:
        print(f"加载检查点: {args.checkpoint}")
        trainer.load_checkpoint(args.checkpoint)
    
    print("=" * 50)
    
    # 开始训练
    try:
        print("开始训练...")
        trainer.train(train_dataloader, val_dataloader, start_epoch=args.start_epoch)
    except KeyboardInterrupt:
        print("\n训练被中断")
        # 保存当前状态
        print("保存当前模型状态...")
        trainer.save_checkpoint(trainer.current_epoch)
        print("模型状态已保存")
    except Exception as e:
        print(f"\n训练过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 50)
    print("训练结束")
    print("=" * 50)