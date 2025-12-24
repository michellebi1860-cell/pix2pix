# Cityscapes数据集加载器
import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class CityscapesDataset(Dataset):
    """Cityscapes数据集加载器"""
    def __init__(self, root_dir, split='train', transform=None):
        """初始化
        参数：
            root_dir: 数据集根目录
            split: 数据集划分（train, val, test）
            transform: 数据变换
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        
        # 获取所有图像路径
        self.image_paths = self._get_image_paths()
        
        # 假设图像和标签是成对存在的
        # 这里需要根据实际的文件命名规则进行调整
        # 例如：图像文件为1.jpg，标签文件为1_label.png
        # 或者图像和标签是同一文件中的左右两部分
        # 由于用户的数据集结构是扁平化的，这里我们假设只有图像文件
        # 或者图像和标签是同一个文件
        # 这里需要根据实际情况进行调整
        # 目前先按照只有图像文件处理
        self.label_paths = self.image_paths
        
    def _get_image_paths(self):
        """获取图像路径"""
        # 数据集目录结构：root_dir/split/*.jpg
        image_dir = os.path.join(self.root_dir, self.split)
        
        image_paths = []
        
        # 获取该目录下所有图像文件
        for filename in os.listdir(image_dir):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # 图像路径
                image_path = os.path.join(image_dir, filename)
                image_paths.append(image_path)
        
        # 排序以确保顺序一致
        image_paths.sort()
        
        return image_paths
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        """获取单个样本
        参数：
            idx: 样本索引
        返回：
            预处理后的图像和标签
        """
        # 加载图像
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # 这里需要根据实际情况处理标签
        # 假设用户的数据集是图像到图像的转换
        # 或者标签是从图像中提取的
        # 目前先将图像作为标签处理
        label = image.copy()
        
        # 应用变换
        if self.transform:
            image, label = self.transform(image, label)
        
        return {'image': image, 'label': label}

class CityscapesTransform:
    """Cityscapes数据集数据变换"""
    def __init__(self, image_size=256):
        """初始化
        参数：
            image_size: 输出图像大小
        """
        self.image_size = image_size
        
    def __call__(self, image, label):
        """应用变换
        参数：
            image: 输入图像
            label: 输入标签
        返回：
            变换后的图像和标签
        """
        # 转换为张量
        image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
        ])
        
        label_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 归一化到[-1, 1]
        ])
        
        transformed_image = image_transform(image)
        transformed_label = label_transform(label)
        
        return transformed_image, transformed_label

if __name__ == '__main__':
    # 测试数据加载器
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    dataset_root = os.path.join(project_root, 'datasets', 'cityscapes')
    
    print(f"Testing dataset at: {dataset_root}")
    
    transform = CityscapesTransform(image_size=256)
    dataset = CityscapesDataset(
        root_dir=dataset_root,
        split='train',
        transform=transform
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # 查看第一个样本
    for sample in dataloader:
        image = sample['image']
        label = sample['label']
        
        print(f"Image shape: {image.shape}")
        print(f"Label shape: {label.shape}")
        break