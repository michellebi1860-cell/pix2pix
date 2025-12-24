# 配置文件
import torch
import os

class Config:
    def __init__(self):
        # 设备配置
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 数据集配置
        self.dataset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'datasets', 'cityscapes')
        self.image_size = 256
        self.num_classes = 35
        
        # 模型配置
        self.gen_in_channels = 3
        self.gen_out_channels = 3
        self.gen_features = 64
        
        self.dis_in_channels = 6  # 3 + 3
        self.dis_features = 64
        self.dis_patch_size = (16, 16)
        
        # 训练配置
        self.batch_size = 2
        self.num_epochs = 20
        self.lr = 0.0002
        self.beta1 = 0.5
        self.beta2 = 0.999
        
        # 损失函数权重
        self.lambda_l1 = 100
        
        # 日志和保存配置
        self.checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
        self.log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
        self.sample_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'samples')
        self.save_epoch_freq = 1
        self.sample_epoch_freq = 1
        
        # 测试配置
        self.test_batch_size = 1
        # 是否在训练结束后计算 FID（而不是每个 epoch）
        self.compute_fid_after_training = True
        
config = Config()