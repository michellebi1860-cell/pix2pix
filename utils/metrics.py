# 评价指标模块
import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import models

class MetricsCalculator:
    """评价指标计算器
    计算Pix2Pix模型生成图像的各种评价指标
    """
    def __init__(self, device='cuda'):
        """初始化
        参数：
            device: 计算设备
        """
        self.device = device
        
        # 初始化用于计算FID的InceptionV3模型
        self.inception_model = self._get_inception_model()
        
    def _get_inception_model(self):
        """获取用于计算FID的InceptionV3模型"""
        # 加载预训练的InceptionV3模型
        inception_model = models.inception_v3(pretrained=True, init_weights=False)
        
        # 修改模型以适应不同的输入尺寸
        # InceptionV3默认输入尺寸为(299, 299)
        inception_model.transform_input = False
        
        # 移除最后的全连接层，保留特征提取部分
        class InceptionFeatureExtractor(torch.nn.Module):
            def __init__(self, model):
                super(InceptionFeatureExtractor, self).__init__()
                self.features = torch.nn.Sequential(
                    model.Conv2d_1a_3x3,
                    model.Conv2d_2a_3x3,
                    model.Conv2d_2b_3x3,
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    model.Conv2d_3b_1x1,
                    model.Conv2d_4a_3x3,
                    torch.nn.MaxPool2d(kernel_size=3, stride=2),
                    model.Mixed_5b,
                    model.Mixed_5c,
                    model.Mixed_5d,
                    model.Mixed_6a,
                    model.Mixed_6b,
                    model.Mixed_6c,
                    model.Mixed_6d,
                    model.Mixed_6e,
                    model.Mixed_7a,
                    model.Mixed_7b,
                    model.Mixed_7c,
                    torch.nn.AdaptiveAvgPool2d((1, 1))
                )
                
            def forward(self, x):
                return self.features(x).view(x.size(0), -1)
        
        feature_extractor = InceptionFeatureExtractor(inception_model)
        feature_extractor = feature_extractor.to(self.device)
        feature_extractor.eval()
        
        return feature_extractor
        
    def calculate_psnr(self, generated_images, real_images):
        """计算峰值信噪比（PSNR）
        参数：
            generated_images: 生成图像张量 (N, C, H, W)
            real_images: 真实图像张量 (N, C, H, W)
        返回：
            平均PSNR值
        """
        psnr_values = []
        
        # 将张量转换为numpy数组，并反归一化
        generated_np = generated_images.detach().cpu().numpy()
        real_np = real_images.detach().cpu().numpy()
        
        # 反归一化：从[-1, 1]转换为[0, 255]
        generated_np = (generated_np + 1) / 2 * 255
        real_np = (real_np + 1) / 2 * 255
        
        # 转换为uint8
        generated_np = generated_np.astype(np.uint8)
        real_np = real_np.astype(np.uint8)
        
        # 计算每个图像的PSNR
        for i in range(generated_np.shape[0]):
            # 转换为(H, W, C)格式
            gen_img = generated_np[i].transpose(1, 2, 0)
            real_img = real_np[i].transpose(1, 2, 0)
            
            psnr = peak_signal_noise_ratio(real_img, gen_img, data_range=255)
            psnr_values.append(psnr)
        
        return np.mean(psnr_values)
        
    def calculate_ssim(self, generated_images, real_images):
        """计算结构相似性（SSIM）
        参数：
            generated_images: 生成图像张量 (N, C, H, W)
            real_images: 真实图像张量 (N, C, H, W)
        返回：
            平均SSIM值
        """
        ssim_values = []
        
        # 将张量转换为numpy数组，并反归一化
        generated_np = generated_images.detach().cpu().numpy()
        real_np = real_images.detach().cpu().numpy()
        
        # 反归一化：从[-1, 1]转换为[0, 255]
        generated_np = (generated_np + 1) / 2 * 255
        real_np = (real_np + 1) / 2 * 255
        
        # 转换为uint8
        generated_np = generated_np.astype(np.uint8)
        real_np = real_np.astype(np.uint8)
        
        # 计算每个图像的SSIM
        for i in range(generated_np.shape[0]):
            # 转换为(H, W, C)格式
            gen_img = generated_np[i].transpose(1, 2, 0)
            real_img = real_np[i].transpose(1, 2, 0)
            
            # 使用固定的win_size=3，这样可以确保不会超过图像尺寸
            try:
                ssim = structural_similarity(
                    real_img, gen_img, 
                    multichannel=True, data_range=255,
                    win_size=3
                )
                ssim_values.append(ssim)
            except Exception as e:
                # 如果计算失败，记录0
                print(f"计算SSIM时出错: {e}")
                ssim_values.append(0)
        
        # 如果没有有效数据，返回0
        if not ssim_values:
            return 0
        
        return np.mean(ssim_values)
        
    def calculate_mae(self, generated_images, real_images):
        """计算平均绝对误差（MAE）
        参数：
            generated_images: 生成图像张量 (N, C, H, W)
            real_images: 真实图像张量 (N, C, H, W)
        返回：
            MAE值
        """
        return F.l1_loss(generated_images, real_images).item()
        
    def calculate_fid(self, generated_images, real_images):
        """计算Frechet Inception Distance（FID）
        参数：
            generated_images: 生成图像张量 (N, C, H, W)
            real_images: 真实图像张量 (N, C, H, W)
        返回：
            FID值
        """
        # 检查是否存在非有限值
        if not torch.isfinite(generated_images).all() or not torch.isfinite(real_images).all():
            print("图像包含非有限值，跳过FID计算")
            # 打印更详细的诊断信息
            try:
                def _stats(x, name):
                    print(f"{name} stats -> device: {x.device}, dtype: {x.dtype}, min: {torch.min(x).item()}, max: {torch.max(x).item()}, mean: {torch.mean(x).item()}, std: {torch.std(x).item()}, nonfinite: { (~torch.isfinite(x)).sum().item() }")
                _stats(generated_images, 'generated_images')
                _stats(real_images, 'real_images')
            except Exception as e:
                print(f"打印图像诊断信息失败: {e}")
            return torch.tensor(float('nan'))
        
        # 调整图像尺寸以适应InceptionV3模型
        # InceptionV3默认输入尺寸为(299, 299)
        # 将像素从[-1, 1]映射到[0, 1]
        generated = (generated_images + 1.0) / 2.0
        real = (real_images + 1.0) / 2.0

        # 调整图像尺寸以适应InceptionV3模型
        # InceptionV3默认输入尺寸为(299, 299)
        generated_resized = F.interpolate(
            generated, size=(299, 299), mode='bilinear', align_corners=False
        )
        real_resized = F.interpolate(
            real, size=(299, 299), mode='bilinear', align_corners=False
        )

        # 将图像标准化为ImageNet的均值和方差（Inception预训练模型期望）
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        generated_resized = (generated_resized.to(self.device).float() - mean) / std
        real_resized = (real_resized.to(self.device).float() - mean) / std
        
        # 检查调整尺寸后的图像是否存在非有限值
        if not torch.isfinite(generated_resized).all() or not torch.isfinite(real_resized).all():
            print("调整尺寸后图像包含非有限值，跳过FID计算")
            return torch.tensor(float('nan'))
        
        # 计算特征向量
        with torch.no_grad():
            try:
                gen_features = self.inception_model(generated_resized)
                real_features = self.inception_model(real_resized)
            except Exception as e:
                print(f"提取特征向量时出错: {e}")
                return torch.tensor(float('nan'))
        
        # 检查特征向量是否存在非有限值
        if not torch.isfinite(gen_features).all() or not torch.isfinite(real_features).all():
            print("特征向量包含非有限值，跳过FID计算")
            try:
                def _fstats(x, name):
                    print(f"{name} shape: {x.shape}, min: {x.min().item()}, max: {x.max().item()}, mean: {x.mean().item()}, std: {x.std().item()}, nonfinite: { (~torch.isfinite(x)).sum().item() }")
                _fstats(gen_features, 'gen_features')
                _fstats(real_features, 'real_features')
            except Exception as e:
                print(f"打印特征诊断信息失败: {e}")
            return torch.tensor(float('nan'))
        
        # 计算均值和协方差
        try:
            gen_mean = torch.mean(gen_features, dim=0)
            gen_cov = self._calculate_covariance(gen_features)
            
            real_mean = torch.mean(real_features, dim=0)
            real_cov = self._calculate_covariance(real_features)
        except Exception as e:
            print(f"计算均值和协方差时出错: {e}")
            return torch.tensor(float('nan'))
        
        # 计算FID
        try:
            fid = self._calculate_frechet_distance(gen_mean, gen_cov, real_mean, real_cov)
        except Exception as e:
            print(f"计算Frechet距离时出错: {e}")
            return torch.tensor(float('nan'))
        
        return fid

    def extract_features(self, images):
        """提取Inception特征向量（用于批次级别的累积）
        参数：
            images: 张量 (N, C, H, W)，值域假定为 [-1, 1]
        返回：
            特征张量 (N, D)（在 CPU 上以节省 GPU 内存）
        """
        if images is None or images.size(0) == 0:
            return torch.empty(0)

        # 映射到 [0,1]
        imgs = (images + 1.0) / 2.0

        # 诊断：检查输入图像是否包含非有限值
        if not torch.isfinite(imgs).all():
            print("extract_features: 输入图像包含非有限值")
            try:
                print(f"images min/max/mean/nonfinite -> min: {torch.min(images).item()}, max: {torch.max(images).item()}, mean: {torch.mean(images).item()}, nonfinite: {(~torch.isfinite(images)).sum().item()}")
            except Exception as e:
                print(f"打印 extract_features 输入诊断失败: {e}")
            # 返回空张量表示不可用
            return torch.empty(0)

        # 调整大小并标准化到 ImageNet 分布
        imgs_resized = F.interpolate(imgs, size=(299, 299), mode='bilinear', align_corners=False)
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        imgs_norm = (imgs_resized.to(self.device).float() - mean) / std

        with torch.no_grad():
            feats = self.inception_model(imgs_norm)

        # 将特征移到 CPU，方便后续在 Trainer 中累积
        return feats.cpu()

    def calculate_fid_from_features(self, gen_features, real_features):
        """基于预先计算的 Inception 特征计算 FID
        参数：
            gen_features: (N, D) 特征张量（CPU 或 device）
            real_features: (M, D) 特征张量（CPU 或 device）
        返回：
            float FID 值，遇到问题返回 None
        """
        try:
            if not isinstance(gen_features, torch.Tensor) or not isinstance(real_features, torch.Tensor):
                print("FID: 特征必须为 torch.Tensor")
                return None

            # 将特征移回 device 以利用已实现的方法
            gen = gen_features.to(self.device)
            real = real_features.to(self.device)

            # 过滤掉包含非有限值的样本行（逐样本检查）
            def _filter_nonfinite(x, name):
                if x.numel() == 0:
                    return x
                finite_mask = torch.isfinite(x).all(dim=1)
                num_total = x.size(0)
                num_finite = finite_mask.sum().item()
                if num_finite != num_total:
                    print(f"FID: 在{name}中发现 {num_total - num_finite} 个包含非有限值的样本，已移除")
                return x[finite_mask]

            gen = _filter_nonfinite(gen, 'gen_features')
            real = _filter_nonfinite(real, 'real_features')

            if gen.size(0) < 2 or real.size(0) < 2:
                print("FID: 样本数量不足，至少需要 2 个有效样本")
                return None

            # 计算均值与协方差
            gen_mean = torch.mean(gen, dim=0)
            gen_cov = self._calculate_covariance(gen)

            real_mean = torch.mean(real, dim=0)
            real_cov = self._calculate_covariance(real)

            # 检查返回的协方差矩阵形状
            if not (isinstance(gen_cov, torch.Tensor) and gen_cov.dim() == 2 and gen_cov.size(0) == gen_cov.size(1)):
                print(f"FID: gen_cov 形状不正确或不可用: {getattr(gen_cov, 'shape', None)}")
                return None
            if not (isinstance(real_cov, torch.Tensor) and real_cov.dim() == 2 and real_cov.size(0) == real_cov.size(1)):
                print(f"FID: real_cov 形状不正确或不可用: {getattr(real_cov, 'shape', None)}")
                return None

            if (not torch.isfinite(gen_mean).all() or not torch.isfinite(real_mean).all() or
                    not torch.isfinite(gen_cov).all() or not torch.isfinite(real_cov).all()):
                print("FID: 特征包含非有限值，跳过计算")
                return None

            fid = self._calculate_frechet_distance(gen_mean, gen_cov, real_mean, real_cov)
            if not torch.isfinite(fid):
                return None
            return float(fid.item())
        except Exception as e:
            print(f"FID 计算出错: {e}")
            return None
        
    def _calculate_covariance(self, features):
        """计算特征向量的协方差矩阵
        参数：
            features: 特征向量张量 (N, D)
        返回：
            协方差矩阵张量 (D, D)
        """
        # 检查特征向量是否存在非有限值
        if not torch.isfinite(features).all():
            print("特征向量包含非有限值，无法计算协方差矩阵")
            return torch.tensor(float('nan'))
            
        n_samples = features.shape[0]
        dim = features.shape[1] if features.dim() >= 2 else 0
        if n_samples < 2:
            print("样本数量不足，无法计算协方差矩阵")
            # 返回一个维度匹配的 NaN 方阵，便于后续检测维度问题
            if dim > 0:
                nan_mat = torch.full((dim, dim), float('nan'))
                return nan_mat
            else:
                return torch.tensor(float('nan'))
            
        # 减去均值
        mean = torch.mean(features, dim=0, keepdim=True)
        features_centered = features - mean
        
        # 检查中心化后的特征向量是否存在非有限值
        if not torch.isfinite(features_centered).all():
            print("中心化后特征向量包含非有限值，无法计算协方差矩阵")
            return torch.tensor(float('nan'))
            
        # 计算协方差矩阵
        covariance = torch.mm(features_centered.t(), features_centered) / (n_samples - 1)
        
        # 检查协方差矩阵是否存在非有限值
        if not torch.isfinite(covariance).all():
            print("协方差矩阵包含非有限值")
            return torch.tensor(float('nan'))
            
        return covariance
        
    def _calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """计算Frechet距离
        参数：
            mu1: 第一个分布的均值 (D)
            sigma1: 第一个分布的协方差矩阵 (D, D)
            mu2: 第二个分布的均值 (D)
            sigma2: 第二个分布的协方差矩阵 (D, D)
            eps: 防止数值不稳定的小值
        返回：
            Frechet距离值 (标量)
        """
        # 检查输入参数是否存在非有限值
        if not torch.isfinite(mu1).all() or not torch.isfinite(sigma1).all() or \
           not torch.isfinite(mu2).all() or not torch.isfinite(sigma2).all():
            print("输入参数包含非有限值，无法计算Frechet距离")
            try:
                print(f"mu1 shape: {mu1.shape}, sigma1 shape: {getattr(sigma1, 'shape', None)}")
                print(f"mu2 shape: {mu2.shape}, sigma2 shape: {getattr(sigma2, 'shape', None)}")
            except Exception:
                pass
            return torch.tensor(float('nan'))
            
        # 计算均值之间的差异
        mu_diff = mu1 - mu2
        
        # 检查均值差异是否存在非有限值
        if not torch.isfinite(mu_diff).all():
            print("均值差异包含非有限值，无法计算Frechet距离")
            return torch.tensor(float('nan'))
            
        # 检查协方差是否为矩阵（2D 方阵）
        if sigma1.dim() != 2 or sigma2.dim() != 2:
            print(f"协方差矩阵维度不正确: sigma1.dim()={getattr(sigma1,'dim', lambda:None)()}, sigma2.dim()={getattr(sigma2,'dim', lambda:None)()}")
            print(f"sigma1 shape: {getattr(sigma1,'shape', None)}, sigma2 shape: {getattr(sigma2,'shape', None)}")
            return torch.tensor(float('nan'))

        # 计算协方差矩阵的和
        sigma_sum = sigma1 + sigma2
        
        # 检查协方差和是否存在非有限值
        if not torch.isfinite(sigma_sum).all():
            print("协方差和包含非有限值，无法计算Frechet距离")
            return torch.tensor(float('nan'))
            
        # 计算平方根矩阵
        # 计算平方根矩阵（多重回退策略以提高稳定性）
        prod = sigma1.mm(sigma2)

        if prod.ndim != 2:
            print(f"计算Frechet距离时出错: prod 不是矩阵，维度为 {prod.ndim}")
            return torch.tensor(float('nan'))

        sqrt_sigma = None

        # 1) 首先尝试使用 torch.linalg.sqrtm（若可用）
        try:
            sqrt_sigma = torch.linalg.sqrtm(prod)
            if not torch.isfinite(sqrt_sigma).all():
                raise RuntimeError("torch.linalg.sqrtm 返回非有限值")
        except Exception as e_sqrtm:
            # 2) 回退到 SVD 方式
            try:
                u, s, v = torch.svd(prod)
                if not torch.isfinite(u).all() or not torch.isfinite(s).all() or not torch.isfinite(v).all():
                    raise RuntimeError("SVD 结果包含非有限值")
                sqrt_sigma = u.mm(torch.diag(torch.sqrt(s + eps))).mm(v.t())
            except Exception as e_svd:
                # 3) 回退到 scipy.linalg.sqrtm（在 CPU 上执行）
                try:
                    import scipy.linalg as la
                    prod_np = prod.cpu().numpy()
                    sqrt_np = la.sqrtm(prod_np)
                    # 如果返回复数且虚部较小，取实部；否则失败
                    if np.iscomplexobj(sqrt_np):
                        max_imag = np.max(np.abs(np.imag(sqrt_np)))
                        if max_imag < 1e-3:
                            sqrt_np = np.real(sqrt_np)
                        else:
                            print(f"计算Frechet距离时出错: scipy.sqrtm 返回较大虚部 ({max_imag})，无法稳定化")
                            return torch.tensor(float('nan'))
                    sqrt_sigma = torch.from_numpy(sqrt_np).to(prod.device).type(prod.dtype)
                except Exception as e_scipy:
                    print(f"计算奇异值分解时出错: {e_svd}; scipy 回退出错: {e_scipy}; 初始 sqrtm 错误: {e_sqrtm}")
                    return torch.tensor(float('nan'))
        
        # 检查平方根矩阵是否存在非有限值
        if not torch.isfinite(sqrt_sigma).all():
            print("平方根矩阵包含非有限值，无法计算Frechet距离")
            return torch.tensor(float('nan'))
            
        # 计算Frechet距离
        # 计算 mu_diff 的平方项（支持 1D 向量或 2D 矩阵）
        try:
            if mu_diff.dim() == 1:
                mu_term = torch.dot(mu_diff, mu_diff)
            else:
                mu_term = mu_diff.t().mm(mu_diff).squeeze()
        except Exception:
            # 作为退路，展平并做点乘
            mu_flat = mu_diff.view(-1)
            mu_term = torch.dot(mu_flat, mu_flat)

        # 确保 sqrt_sigma 为 2D 矩阵
        if sqrt_sigma is None or sqrt_sigma.dim() != 2:
            print(f"计算Frechet距离时出错: sqrt_sigma 不是 2D 矩阵，类型/形状: {getattr(sqrt_sigma, 'shape', None)}")
            return torch.tensor(float('nan'))

        fid = mu_term + torch.trace(sigma1) + torch.trace(sigma2) - 2 * torch.trace(sqrt_sigma)
        
        # 检查FID值是否存在非有限值
        if not torch.isfinite(fid):
            print("FID值包含非有限值")
            return torch.tensor(float('nan'))
            
        return fid
        
    def calculate_all_metrics(self, generated_images, real_images):
        """计算所有评价指标
        参数：
            generated_images: 生成图像张量 (N, C, H, W)
            real_images: 真实图像张量 (N, C, H, W)
        返回：
            包含所有指标的字典
        """
        metrics = {}
        
        metrics['psnr'] = self.calculate_psnr(generated_images, real_images)
        metrics['ssim'] = self.calculate_ssim(generated_images, real_images)
        metrics['mae'] = self.calculate_mae(generated_images, real_images)
        
        # 计算FID（可能返回NaN，遇到NaN则记为None）
        try:
            fid_val = self.calculate_fid(generated_images, real_images)
            if isinstance(fid_val, torch.Tensor):
                if not torch.isfinite(fid_val):
                    metrics['fid'] = None
                else:
                    metrics['fid'] = float(fid_val.item())
            else:
                metrics['fid'] = None
        except Exception:
            metrics['fid'] = None
        
        return metrics

if __name__ == '__main__':
    # 测试评价指标计算器
    import torch
    
    # 创建测试数据
    batch_size = 2
    generated_images = torch.randn(batch_size, 3, 256, 256)
    real_images = torch.randn(batch_size, 3, 256, 256)
    
    # 创建指标计算器
    calculator = MetricsCalculator(device='cpu')
    
    # 计算所有指标
    metrics = calculator.calculate_all_metrics(generated_images, real_images)
    
    print("评价指标:")
    for key, value in metrics.items():
        print(f"  {key}: {value}")