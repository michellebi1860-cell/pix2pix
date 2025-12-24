# 训练循环模块
import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import datetime

class Trainer:
    """Pix2Pix模型训练器"""
    def __init__(self, config, generator, discriminator, loss_fn, metrics_calculator):
        """初始化
        参数：
            config: 配置对象
            generator: 生成器模型
            discriminator: 判别器模型
            loss_fn: 损失函数对象
            metrics_calculator: 评价指标计算器对象
        """
        self.config = config
        self.device = config.device
        
        # 模型
        self.generator = generator.to(self.device)
        self.discriminator = discriminator.to(self.device)
        
        # 损失函数
        self.loss_fn = loss_fn
        
        # 评价指标计算器
        self.metrics_calculator = metrics_calculator
        
        # 优化器
        self.generator_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )
        
        self.discriminator_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=config.lr,
            betas=(config.beta1, config.beta2)
        )
        
        # 创建保存目录
        self._create_directories()
        
        # 日志
        self.start_time = datetime.datetime.now()
        
    def _create_directories(self):
        """创建必要的目录"""
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        os.makedirs(self.config.log_dir, exist_ok=True)
        os.makedirs(self.config.sample_dir, exist_ok=True)
        
    def load_checkpoint(self, checkpoint_path):
        """加载模型检查点
        参数：
            checkpoint_path: 检查点文件路径
        """
        if os.path.exists(checkpoint_path):
            print(f"加载检查点: {checkpoint_path}")
            
            checkpoint = torch.load(checkpoint_path)
            
            # 加载模型权重
            self.generator.load_state_dict(checkpoint['generator_state_dict'])
            self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
            
            # 加载优化器状态
            self.generator_optimizer.load_state_dict(checkpoint['generator_optimizer_state_dict'])
            self.discriminator_optimizer.load_state_dict(checkpoint['discriminator_optimizer_state_dict'])
            
            # 更新学习率（如果需要）
            for param_group in self.generator_optimizer.param_groups:
                param_group['lr'] = self.config.lr
            
            for param_group in self.discriminator_optimizer.param_groups:
                param_group['lr'] = self.config.lr
            
            print(f"检查点加载完成")
            
        else:
            print(f"检查点不存在: {checkpoint_path}")
            
    def save_checkpoint(self, epoch):
        """保存模型检查点
        参数：
            epoch: 当前训练轮次
        """
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_epoch_{epoch}.pth"
        )
        
        torch.save({
            'epoch': epoch,
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'generator_optimizer_state_dict': self.generator_optimizer.state_dict(),
            'discriminator_optimizer_state_dict': self.discriminator_optimizer.state_dict(),
            'config': self.config,
        }, checkpoint_path)
        
        print(f"检查点保存至: {checkpoint_path}")
        
    def save_sample(self, epoch, labels, real_images, generated_images):
        """保存验证集样本的三联图（标签、生成图像、真实图像）
        参数：
            epoch: 当前训练轮次
            labels: 语义标签图像
            real_images: 真实图像
            generated_images: 生成图像
        """
        # 选择第一个样本
        label_sample = labels[0]
        real_sample = real_images[0]
        generated_sample = generated_images[0]
        
        # 将标签转换为可视化格式
        # 简单处理：取最大值所在的通道作为类别
        label_visual = torch.argmax(label_sample, dim=0, keepdim=True)
        # 归一化到[0, 1]
        label_visual = label_visual.float() / label_visual.max()
        # 扩展到3个通道
        label_visual = label_visual.expand(3, -1, -1)
        
        # 拼接三个图像
        sample = torch.cat([label_visual, generated_sample, real_sample], dim=2)
        
        # 保存图像
        save_image_path = os.path.join(
            self.config.sample_dir,
            f"sample_epoch_{epoch}.png"
        )
        
        save_image(sample, save_image_path, normalize=True)
        print(f"样本保存至: {save_image_path}")
        
    def train_epoch(self, dataloader):
        """训练一个epoch
        参数：
            dataloader: 训练数据加载器
        返回：
            平均损失和指标
        """
        self.generator.train()
        self.discriminator.train()
        
        epoch_losses = {}
        epoch_metrics = {}
        
        for i, batch in enumerate(dataloader):
            # 获取数据
            real_images = batch['image'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # ------------------
            # 训练生成器
            # ------------------
            self.generator_optimizer.zero_grad()
            
            # 生成图像
            generated_images = self.generator(labels)
            
            # 计算生成器损失
            losses = self.loss_fn(self.discriminator, real_images, generated_images, labels)
            generator_loss = losses['generator_total_loss']
            # 反向传播（添加数值稳健性检查）
            if not isinstance(generator_loss, torch.Tensor):
                print("Generator loss is not a tensor, skipping backward")
            else:
                if not torch.isfinite(generator_loss).all():
                    print("Generator loss 包含非有限值，跳过该 step 并打印诊断信息")
                    try:
                        print(f"generator_loss: {generator_loss}")
                        print(f"generated_images nonfinite: {(~torch.isfinite(generated_images)).sum().item()} / {generated_images.numel()}")
                        print(f"generated_images min/max/mean: {generated_images.min().item()}/{generated_images.max().item()}/{generated_images.mean().item()}")
                        print(f"real_images nonfinite: {(~torch.isfinite(real_images)).sum().item()} / {real_images.numel()}")
                    except Exception as e:
                        print(f"打印诊断信息失败: {e}")
                else:
                    generator_loss.backward()

                    # 梯度裁剪以防止梯度爆炸
                    try:
                        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=5.0)
                    except Exception:
                        pass

                    # 检查梯度是否包含非有限值
                    grad_finite = True
                    for p in self.generator.parameters():
                        if p.grad is not None and (not torch.isfinite(p.grad).all()):
                            grad_finite = False
                            print("检测到 generator grad 包含非有限值，跳过 optimizer.step()")
                            break

                    if grad_finite:
                        self.generator_optimizer.step()
                    else:
                        # 清理梯度以防止影响后续步骤
                        self.generator_optimizer.zero_grad()
            
            # ------------------
            # 训练判别器
            # ------------------
            self.discriminator_optimizer.zero_grad()
            
            # 计算判别器损失
            losses = self.loss_fn(self.discriminator, real_images, generated_images.detach(), labels)
            discriminator_loss = losses['discriminator_total_loss']

            # 反向传播（仅当判别器损失是计算图的一部分时才更新判别器）
            # 这可以避免在仅使用L1损失时对一个常数张量调用 backward 导致的错误
            if isinstance(discriminator_loss, torch.Tensor) and discriminator_loss.requires_grad:
                discriminator_loss.backward()

                # 梯度裁剪
                try:
                    torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=5.0)
                except Exception:
                    pass

                # 检查判别器梯度是否包含非有限值
                grad_finite = True
                for p in self.discriminator.parameters():
                    if p.grad is not None and (not torch.isfinite(p.grad).all()):
                        grad_finite = False
                        print("检测到 discriminator grad 包含非有限值，跳过 optimizer.step()")
                        break

                if grad_finite:
                    self.discriminator_optimizer.step()
                else:
                    self.discriminator_optimizer.zero_grad()
            else:
                # 跳过判别器更新（例如 L1OnlyLoss 返回的常数零损失）
                pass
            
            # 记录损失
            for key, value in losses.items():
                if key not in epoch_losses:
                    epoch_losses[key] = []
                epoch_losses[key].append(value.item())
            
            # 记录指标
            metrics = self.metrics_calculator.calculate_all_metrics(generated_images, real_images)
            for key, value in metrics.items():
                if key not in epoch_metrics:
                    epoch_metrics[key] = []
                if value is not None:
                    epoch_metrics[key].append(value)
            
            # 打印训练进度
            if (i + 1) % 100 == 0:
                print(f"  Batch {i + 1}/{len(dataloader)}: ")
                print(f"    Generator Loss: {losses['generator_total_loss'].item():.4f}")
                print(f"    Discriminator Loss: {losses['discriminator_total_loss'].item():.4f}")
                print(f"    PSNR: {metrics.get('psnr', 0):.4f}, SSIM: {metrics.get('ssim', 0):.4f}")
        
        # 计算平均损失和指标
        avg_epoch_losses = {key: np.mean(value) for key, value in epoch_losses.items()}
        avg_epoch_metrics = {key: np.mean(value) for key, value in epoch_metrics.items()}
        
        return avg_epoch_losses, avg_epoch_metrics
        
    def validate(self, dataloader):
        """验证模型
        参数：
            dataloader: 验证数据加载器
        返回：
            平均损失和指标
        """
        self.generator.eval()
        self.discriminator.eval()
        
        val_losses = {}
        val_metrics = {}

        # FID 在验证阶段不再计算（将在训练结束后一次性计算）
        
        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                # 获取数据
                real_images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # 生成图像
                generated_images = self.generator(labels)
                
                # 计算损失
                losses = self.loss_fn(self.discriminator, real_images, generated_images, labels)
                
                # 记录损失
                for key, value in losses.items():
                    if key not in val_losses:
                        val_losses[key] = []
                    val_losses[key].append(value.item())
                
                # 记录指标
                metrics = self.metrics_calculator.calculate_all_metrics(generated_images, real_images)
                for key, value in metrics.items():
                    if key not in val_metrics:
                        val_metrics[key] = []
                    if value is not None:
                        val_metrics[key].append(value)

                # （已禁用）在验证阶段不再累积 Inception 特征以计算 FID
                
                # 保存第一个样本用于可视化
                if i == 0:
                    self.save_sample(self.current_epoch, labels, real_images, generated_images)
        
        # 计算平均损失和指标
        avg_val_losses = {key: np.mean(value) for key, value in val_losses.items()}
        avg_val_metrics = {key: np.mean(value) for key, value in val_metrics.items()}

        # 不在验证阶段计算 FID（将在训练结束后一次性计算）
        avg_val_metrics['fid'] = None
        
        return avg_val_losses, avg_val_metrics

    def compute_fid_on_dataloader(self, dataloader):
        """在给定 dataloader 上累积 Inception 特征并计算 FID（在训练结束后调用）
        返回：float FID 值或 None
        """
        self.generator.eval()

        gen_feats_list = []
        real_feats_list = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                real_images = batch['image'].to(self.device)
                labels = batch['label'].to(self.device)

                # 生成图像
                generated_images = self.generator(labels)

                # 提取特征（返回 CPU 张量 或 空张量）
                try:
                    gen_feats = self.metrics_calculator.extract_features(generated_images)
                    real_feats = self.metrics_calculator.extract_features(real_images)
                    if isinstance(gen_feats, torch.Tensor) and gen_feats.numel() > 0:
                        gen_feats_list.append(gen_feats)
                    if isinstance(real_feats, torch.Tensor) and real_feats.numel() > 0:
                        real_feats_list.append(real_feats)
                except Exception as e:
                    print(f"在计算训练后 FID 时提取特征出错（batch {i}）: {e}")

        if not gen_feats_list or not real_feats_list:
            print("计算训练后 FID: 没有有效的特征可用于计算 FID")
            return None

        gen_feats_all = torch.cat(gen_feats_list, dim=0)
        real_feats_all = torch.cat(real_feats_list, dim=0)

        # 诊断输出
        try:
            print(f"训练后累积特征 - gen shape: {gen_feats_all.shape}, real shape: {real_feats_all.shape}")
            print(f"gen nonfinite rows: {(~torch.isfinite(gen_feats_all)).any(dim=1).sum().item()} / {gen_feats_all.size(0)}")
            print(f"real nonfinite rows: {(~torch.isfinite(real_feats_all)).any(dim=1).sum().item()} / {real_feats_all.size(0)}")
        except Exception:
            pass

        fid_val = self.metrics_calculator.calculate_fid_from_features(gen_feats_all, real_feats_all)
        return fid_val
        
    def train(self, train_dataloader, val_dataloader, start_epoch=0):
        """训练模型
        参数：
            train_dataloader: 训练数据加载器
            val_dataloader: 验证数据加载器
            start_epoch: 开始训练的轮次
        """
        print(f"开始训练，从第 {start_epoch} 轮开始")
        print(f"训练集大小: {len(train_dataloader.dataset)}")
        print(f"验证集大小: {len(val_dataloader.dataset)}")
        print(f"设备: {self.device}")
        print(f"开始时间: {self.start_time}")
        print("=" * 50)
        
        for epoch in range(start_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # 训练一个epoch
            print("训练阶段:")
            train_losses, train_metrics = self.train_epoch(train_dataloader)
            
            # 验证模型
            print("\n验证阶段:")
            val_losses, val_metrics = self.validate(val_dataloader)
            
            # 打印训练和验证结果
            print("\n训练结果:")
            for key, value in train_losses.items():
                print(f"  {key}: {value:.4f}")
            for key, value in train_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            print("\n验证结果:")
            for key, value in val_losses.items():
                print(f"  {key}: {value:.4f}")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")
            
            # 保存模型检查点
            if (epoch + 1) % self.config.save_epoch_freq == 0:
                self.save_checkpoint(epoch + 1)
            
            # 打印训练时间
            elapsed_time = datetime.datetime.now() - self.start_time
            print(f"\n已用时间: {elapsed_time}")
            print("=" * 50)
        # 训练循环结束

        # 在训练结束后计算 FID（如果配置启用）
        if getattr(self.config, 'compute_fid_after_training', False):
            try:
                print("开始在完整验证集上计算 FID...")
                fid_val = self.compute_fid_on_dataloader(val_dataloader)
                print(f"训练结束后验证集 FID: {fid_val}")
            except Exception as e:
                print(f"计算训练后 FID 出错: {e}")

        print("\n训练完成！")
        print(f"总训练时间: {datetime.datetime.now() - self.start_time}")
        print(f"最终模型检查点已保存至: {self.config.checkpoint_dir}")
        print(f"样本图像已保存至: {self.config.sample_dir}")