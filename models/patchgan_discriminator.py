# PatchGAN判别器网络
import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    def __init__(self, in_channels, features=64):
        super(PatchGANDiscriminator, self).__init__()
        
        # 判别器网络结构
        self.disc = nn.Sequential(
            # 第一层：不使用归一化
            nn.Conv2d(
                in_channels, features, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 第二层
            self._conv_block(features, features * 2),
            
            # 第三层
            self._conv_block(features * 2, features * 4),
            
            # 第四层：使用更小的步长
            nn.Conv2d(
                features * 4, features * 8, kernel_size=4, stride=1, padding=1
            ),
            nn.BatchNorm2d(features * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 最终层：输出Patch级别的判断
            nn.Conv2d(
                features * 8, 1, kernel_size=4, stride=1, padding=1
            ),
            nn.Sigmoid()
        )
        
    def _conv_block(self, in_channels, out_channels):
        """卷积块：Conv2d -> BatchNorm2d -> LeakyReLU"""
        return nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel_size=4, stride=2, padding=1
            ),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x, y):
        """前向传播
        参数：
            x: 输入图像（语义标签）
            y: 目标图像（真实图像或生成图像）
        返回：
            每个Patch的真伪判断
        """
        # 将输入图像和目标图像拼接在一起
        input = torch.cat((x, y), dim=1)
        return self.disc(input)