import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class RoadDataset(Dataset):
    def __init__(self, root_path, dataset_name, mode='train', img_size=1024):
        self.mode = mode
        self.dataset_name = dataset_name  # 🌟 新增：把数据集的名字保存下来！
        self.img_dir = os.path.join(root_path, dataset_name, mode, 'images')
        self.mask_dir = os.path.join(root_path, dataset_name, mode, 'masks')
        self.img_list = os.listdir(self.img_dir)
        
        # 在 __init__ 里，替换为标准的 ImageNet 均值和方差
        if self.mode == 'train':
            self.transform = A.Compose([
                # 🌟 核心：训练集使用“随机裁剪”！
                # 每次读取时，它会像一个 1024x1024 的放大镜，在 1500x1500 的图上随机框取一块。
                # 这等价于无穷无尽的数据增强，模型每个 Epoch 看到的局部都不一样！
                A.RandomCrop(height=img_size, width=img_size),
                         
                A.HueSaturationValue(hue_shift_limit=30, sat_shift_limit=5, val_shift_limit=15, p=0.5),
                
                # 🌟 核心修改 2：旋转和平移同样会产生插值，也必须严加防范！
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=0, 
                                   interpolation=cv2.INTER_LINEAR, 
                                   mask_interpolation=cv2.INTER_NEAREST,
                                   border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0, p=0.5),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                
                # 🌟 黄金法则：使用 ImageNet 标准归一化！
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225), 
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ])
        else:
            self.transform = A.Compose([
                # 🌟 修正：验证集和测试集使用中心裁剪，保持道路宽度绝对真实，且每次评估结果固定！
                A.CenterCrop(height=img_size, width=img_size),
                # 🌟 验证集也要保持一致！
                A.Normalize(
                    mean=(0.485, 0.456, 0.406), 
                    std=(0.229, 0.224, 0.225), 
                    max_pixel_value=255.0
                ),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img_name = self.img_list[idx]
        img_path = os.path.join(self.img_dir, img_name)

        # 🌟 严谨匹配：根据 JSON 传进来的名字，绝对精准地替换后缀
        if self.dataset_name.lower() == 'deepglobe':
            # 例: 12345_sat.jpg -> 12345_mask.png
            mask_name = img_name.replace('_sat.jpg', '_mask.png')
            
        elif self.dataset_name.lower() == 'massachusetts':
            # 例: 10378780_15.tiff -> 10378780_15.tif
            mask_name = img_name.replace('.tiff', '.tif')
            
        else:
            raise ValueError(f"未知的 Dataset 名字: {self.dataset_name}，请检查 JSON 配置！")

        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # 1. 读取图片 (加了安全锁，防止某张图片损坏导致整个训练崩溃)
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"图片损坏或路径错误，无法读取: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. 读取标签 (灰度图)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"标签损坏或路径错误，无法读取: {mask_path}")
        
        # 3. 通过 Albumentations 同步变换 (图像和掩码严丝合缝)
        transformed = self.transform(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # 4. 标签二值化处理 
        # 只要像素值大于 0 (不管是 1 还是 255)，全都强制变成 1.0；否则是 0.0
        mask = (mask > 0).float()
        mask = mask.unsqueeze(0)     # 增加通道维度 [1, H, W]
        
        return image, mask, img_name