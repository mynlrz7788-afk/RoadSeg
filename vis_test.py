import os
import json
import torch
import cv2
import numpy as np
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataloaders.road_dataset import RoadDataset
from models import get_model

def draw_paper_visualization(img_np, pred_mask, gt_mask):
    """
    生成学术界顶刊通用的“错误分析覆盖图 (Error Analysis Overlay Map)”
    绿色: TP (猜对了) | 红色: FP (猜错了/报假警) | 蓝色: FN (漏掉了/没找出来)
    """
    # 准备一个和原图大小一样的黑色画布 (H, W, 3)
    color_mask = np.zeros_like(img_np)
    
    # 逻辑运算：找出三种状态的具体像素位置
    TP = (pred_mask == 1) & (gt_mask == 1)
    FP = (pred_mask == 1) & (gt_mask == 0)
    FN = (pred_mask == 0) & (gt_mask == 1)

    # 填色提醒：OpenCV 默认的颜色通道顺序是 BGR，不是 RGB！
    color_mask[TP] = [0, 255, 0]    # 绿色 (B=0, G=255, R=0)
    color_mask[FP] = [0, 0, 255]    # 红色 (B=0, G=0, R=255)
    color_mask[FN] = [255, 0, 0]    # 蓝色 (B=255, G=0, R=0)

    # 图像混合 (Image Blending)：将彩色面具像贴膜一样贴在原图上
    alpha = 0.6  # 原图的可见度 (留 60%，防止背景太亮看不清道路)
    beta = 1.0   # 道路颜色的可见度 (100% 高亮)
    overlay_img = cv2.addWeighted(img_np, alpha, color_mask, beta, 0)
    
    return overlay_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-w', '--weight', type=str, required=True)
    # 增加一个控制数量的参数
    parser.add_argument('-n', '--num_imgs', type=int, default=30, help='要生成的论文插图数量')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    val_dataset = RoadDataset(config['dataset']['root_path'], config['dataset']['name'], mode='val', img_size=config['dataset'].get('input_size', 1024))
    
    # 【注意】画图时必须严格将 batch_size 设为 1！否则多张图混在一个 batch 里处理和保存会非常麻烦
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = get_model(config['model']).cuda()
    checkpoint = torch.load(args.weight)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
        
    model.eval()

    # 在权重所在的目录下，新建一个专门放图的文件夹
    save_dir = os.path.dirname(args.weight)
    vis_dir = os.path.join(save_dir, 'paper_visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    print(f"开始生成图，将保存前 {args.num_imgs} 张到: {vis_dir}")

    with torch.no_grad():
        # 解包为 batch_data，为了兼容 Dataset 返回文件名的逻辑
        for i, batch_data in enumerate(tqdm(val_loader, desc="Drawing Figures", total=args.num_imgs)):
            
            # 达到了设定好的画图数量，立刻刹车退出，节约硬盘和时间
            if i >= args.num_imgs:
                break 
                
            # 判断 Dataset 返回了几个变量
            if len(batch_data) == 3:
                imgs, masks, img_paths = batch_data
                # 提取文件名，比如从 "/data/103423_sat.jpg" 提取出 "103423_sat"
                base_name = os.path.splitext(os.path.basename(img_paths[0]))[0]
            else:
                imgs, masks = batch_data
                base_name = f"paper_result_{i}" # 如果没改 dataset，自动退回到用编号命名

            imgs, masks = imgs.cuda(), masks.cuda()
            preds = model(imgs)
            
            # --- 逆向归一化处理 (Un-normalization) ---
            # 因为数据在 DataLoader 里被除以了 255，并减均值除以方差变成了标准化 Tensor
            # 原路退回去
            img_np = imgs[0].cpu().permute(1, 2, 0).numpy()  # CHW 变 HWC (🌟 修复色彩：去掉了这里的 * 255.0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            # 修复色彩：先乘以 std 加上 mean 恢复到 0~1 的实数，再统一乘以 255.0
            img_np = (img_np * std + mean) * 255.0 
            img_np = np.clip(img_np, 0, 255).astype(np.uint8)  # 恢复并裁剪越界像素
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR) # 转给 OpenCV 喜欢的 BGR 通道
            
            # --- 提取模型的 0/1 掩码 ---
            pred_np = (preds[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            gt_np = (masks[0, 0].cpu().numpy() > 0.5).astype(np.uint8)
            
            # 调用函数，生成彩色覆盖图
            overlay_img = draw_paper_visualization(img_np, pred_np, gt_np)
            
            # --- 拼接对比图 (Image Matrix) ---
            # 真实标签原本是单通道 (H, W)，必须转成三通道 (H, W, 3) 才能和彩图横向拼接
            gt_vis = cv2.cvtColor(gt_np * 255, cv2.COLOR_GRAY2BGR) 
            
            # 把模型的 0/1 预测结果，也放大到 255 并转成三通道黑白图
            pred_vis = cv2.cvtColor(pred_np * 255, cv2.COLOR_GRAY2BGR)
            
            # 横向拼接 4 张图：[左侧原图 | 真实黑白标签 | 预测黑白标签 | 右侧彩色分析图]
            concat_img = np.hstack([img_np, gt_vis, pred_vis, overlay_img])
            
            # 保存到硬盘 (使用真实的文件名来保存)
            cv2.imwrite(os.path.join(vis_dir, f"{base_name}_analysis.png"), concat_img)

    print(f"🎉 画图完成！")

if __name__ == '__main__':
    main()