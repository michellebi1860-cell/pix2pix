# 损失函数模块
import torch
import torch.nn as nn

class Pix2PixLoss(nn.Module):
    """Pix2Pix模型损失函数
    组合了L1损失和对抗损失
    """
    def __init__(self, lambda_l1=100):
        """初始化
        参数：
            lambda_l1: L1损失的权重
        """
        super(Pix2PixLoss, self).__init__()
        
        # 对抗损失：二元交叉熵损失
        self.adversarial_loss = nn.BCELoss()
        
        # L1损失：平均绝对误差
        self.l1_loss = nn.L1Loss()
        
        # L1损失权重
        self.lambda_l1 = lambda_l1
        
    def forward(self, discriminator, real_images, generated_images, labels):
        """计算损失
        参数：
            discriminator: 判别器网络
            real_images: 真实图像
            generated_images: 生成图像
            labels: 语义标签
        返回：
            损失字典
        """
        loss_dict = {}
        
        # ------------------
        # 计算生成器损失
        # ------------------
        # 生成器希望判别器将生成的图像判断为真实图像
        valid_labels = torch.ones_like(discriminator(labels, generated_images))
        generator_adversarial_loss = self.adversarial_loss(
            discriminator(labels, generated_images), valid_labels
        )
        
        # 生成器的L1损失：生成图像与真实图像的像素差异
        generator_l1_loss = self.l1_loss(generated_images, real_images)
        
        # 生成器总损失
        generator_loss = generator_adversarial_loss + self.lambda_l1 * generator_l1_loss
        
        # ------------------
        # 计算判别器损失
        # ------------------
        # 判别器希望将真实图像判断为真实
        real_valid_labels = torch.ones_like(discriminator(labels, real_images))
        discriminator_real_loss = self.adversarial_loss(
            discriminator(labels, real_images), real_valid_labels
        )
        
        # 判别器希望将生成图像判断为虚假
        fake_valid_labels = torch.zeros_like(discriminator(labels, generated_images))
        discriminator_fake_loss = self.adversarial_loss(
            discriminator(labels, generated_images.detach()), fake_valid_labels
        )
        
        # 判别器总损失
        discriminator_loss = (discriminator_real_loss + discriminator_fake_loss) / 2
        
        # 存储所有损失
        loss_dict['generator_total_loss'] = generator_loss
        loss_dict['generator_adversarial_loss'] = generator_adversarial_loss
        loss_dict['generator_l1_loss'] = generator_l1_loss
        loss_dict['discriminator_total_loss'] = discriminator_loss
        loss_dict['discriminator_real_loss'] = discriminator_real_loss
        loss_dict['discriminator_fake_loss'] = discriminator_fake_loss
        
        return loss_dict

class L1OnlyLoss(nn.Module):
    """仅使用L1损失的模型
    用于比较L1损失与L1+GAN损失的效果
    """
    def __init__(self):
        """初始化"""
        super(L1OnlyLoss, self).__init__()
        
        # L1损失：平均绝对误差
        self.l1_loss = nn.L1Loss()
        
    def forward(self, discriminator, real_images, generated_images, labels):
        """计算损失
        参数：
            discriminator: 判别器网络（仅为了保持接口一致性，实际未使用）
            real_images: 真实图像
            generated_images: 生成图像
            labels: 语义标签（仅为了保持接口一致性）
        返回：
            损失字典
        """
        loss_dict = {}
        
        # 仅计算L1损失
        generator_loss = self.l1_loss(generated_images, real_images)
        
        # 存储损失（使用与输入张量相同设备/类型的常数，避免设备不匹配）
        zero = generated_images.new_tensor(0.0)
        loss_dict['generator_total_loss'] = generator_loss
        loss_dict['generator_adversarial_loss'] = zero
        loss_dict['generator_l1_loss'] = generator_loss
        loss_dict['discriminator_total_loss'] = zero
        loss_dict['discriminator_real_loss'] = zero
        loss_dict['discriminator_fake_loss'] = zero
        
        return loss_dict

if __name__ == '__main__':
    # 测试损失函数
    from models.unet_generator import UNetGenerator
    from models.patchgan_discriminator import PatchGANDiscriminator
    import torch
    
    # 创建模型
    generator = UNetGenerator(in_channels=35, out_channels=3)
    discriminator = PatchGANDiscriminator(in_channels=38)
    
    # 创建损失函数
    pix2pix_loss = Pix2PixLoss(lambda_l1=100)
    l1_only_loss = L1OnlyLoss()
    
    # 创建测试数据
    batch_size = 1
    labels = torch.randn(batch_size, 35, 256, 256)
    real_images = torch.randn(batch_size, 3, 256, 256)
    generated_images = generator(labels)
    
    # 计算Pix2Pix损失
    pix2pix_losses = pix2pix_loss(discriminator, real_images, generated_images, labels)
    print("Pix2Pix Losses:")
    for key, value in pix2pix_losses.items():
        print(f"  {key}: {value.item()}")
    
    # 计算仅L1损失
    l1_losses = l1_only_loss(discriminator, real_images, generated_images, labels)
    print("\nL1 Only Losses:")
    for key, value in l1_losses.items():
        print(f"  {key}: {value.item()}")