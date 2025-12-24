# U-Net生成器网络
import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels, out_channels, features=64):
        super(UNetGenerator, self).__init__()
        
        # 编码部分（下采样）
        self.encoder = nn.ModuleList([
            self._conv_block(in_channels, features, normalize=False),
            self._conv_block(features, features * 2),
            self._conv_block(features * 2, features * 4),
            self._conv_block(features * 4, features * 8),
            self._conv_block(features * 8, features * 8),
            self._conv_block(features * 8, features * 8),
            self._conv_block(features * 8, features * 8),
        ])
        
        # 解码部分（上采样）
        self.decoder = nn.ModuleList([
            self._up_conv_block(features * 8, features * 8, dropout=True),
            self._up_conv_block(features * 16, features * 8, dropout=True),
            self._up_conv_block(features * 16, features * 8, dropout=True),
            self._up_conv_block(features * 16, features * 4),
            self._up_conv_block(features * 8, features * 2),
            self._up_conv_block(features * 4, features),
        ])
        
        # 最终的卷积层
        self.final_conv = nn.ConvTranspose2d(
            features * 2, out_channels, kernel_size=4, stride=2, padding=1
        )
        self.tanh = nn.Tanh()
        
    def _conv_block(self, in_channels, out_channels, normalize=True):
        """卷积块：Conv2d -> BatchNorm2d -> LeakyReLU"""
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        return nn.Sequential(*layers)
        
    def _up_conv_block(self, in_channels, out_channels, dropout=False):
        """上采样卷积块：ConvTranspose2d -> BatchNorm2d -> Dropout -> ReLU"""
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU(inplace=True))
        return nn.Sequential(*layers)
        
    def forward(self, x):
        """前向传播"""
        # 存储编码部分的输出，用于跳跃连接
        skip_connections = []
        
        # 编码过程
        for down in self.encoder:
            x = down(x)
            skip_connections.append(x)
        
        # 移除最后一个跳跃连接（因为解码部分从最底层开始）
        skip_connections = skip_connections[:-1]
        
        # 解码过程，与跳跃连接合并
        for idx, up in enumerate(self.decoder):
            x = up(x)
            # 从后往前获取跳跃连接
            x = torch.cat((x, skip_connections[-(idx + 1)]), dim=1)
        
        # 最终卷积
        x = self.final_conv(x)
        x = self.tanh(x)
        
        return x