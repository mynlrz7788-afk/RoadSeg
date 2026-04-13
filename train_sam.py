import os
import time
import datetime
import json
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from fvcore.nn import FlopCountAnalysis
from tqdm import tqdm
import torch.amp
# 导入组件
from dataloaders.road_dataset import RoadDataset
from models import get_model
from core.loss import BCEDiceLoss
from core.metrics import Evaluator
def create_experiment_dir(config):
    """自动生成日志文件夹"""
    dataset_name = config['dataset']['name']
    model_name = config['model']['name']
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    
    exp_dir = os.path.join('saved_runs', dataset_name, f"{model_name}_{timestamp}")
    os.makedirs(exp_dir, exist_ok=True)
    
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)
    return exp_dir

def evaluate_model_complexity(model, device, img_size):
    """评估模型的 Params, FLOPs 和 FPS，并返回数值 (基于 fvcore 升级版)"""
    model.eval() 
    dummy_input = torch.randn(1, 3, img_size, img_size).to(device)
    
    # 1. 精准计算参数量 (Params)
    params = sum(p.numel() for p in model.parameters())
    
    # 2. 使用 fvcore 计算计算量 (FLOPs)
    with torch.no_grad():
        flops_analyzer = FlopCountAnalysis(model, dummy_input)
        # 关闭对不支持算子的警告，保持控制台整洁
        flops_analyzer.unsupported_ops_warnings(False)
        flops_analyzer.uncalled_modules_warnings(False)
        
        # fvcore 计算出的是 MACs (乘加操作数)。
        # 学术界通用的换算标准是: 1 MAC = 2 FLOPs，为了和你原本的 thop 兼容，这里直接乘 2
        macs = flops_analyzer.total()
        flops = macs * 2 
    
    # 3. 计算推理速度 (FPS)
    with torch.no_grad():
        # 预热 (必须有，否则第一次显存分配会拉低速度)
        for _ in range(20):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        
        start_time = time.time()
        for _ in range(50):
            _ = model(dummy_input)
        torch.cuda.synchronize()
        end_time = time.time()
        
    fps = 50.0 / (end_time - start_time)
    
    # 恢复训练模式
    model.train()  
    
    return params, flops, fps

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
    parser.add_argument('-r', '--resume', type=str, default=None, help='Path to latest_model.pth to resume')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)

    # 🌟 修改点：判断是否是断点续训
    if args.resume and os.path.isfile(args.resume):
        # 提取原来存放 latest_model.pth 的文件夹路径
        exp_dir = os.path.dirname(args.resume)
        # ⚠️ 注意：这里必须用 'a' (append) 追加模式，否则会清空之前的日志！
        log_file = open(os.path.join(exp_dir, 'train.log'), 'a')
        
        # 可以在日志里打一条明显的分割线，方便区分是哪次续训的
        resume_msg = f"\n{'='*50}\n 🔌 触发断点续训，继续向原目录记录日志 \n{'='*50}\n"
        print(resume_msg, end='')
        log_file.write(resume_msg)
    else:
        # 正常的新实验，创建新文件夹，使用 'w' (write) 模式覆盖写入
        exp_dir = create_experiment_dir(config)
        log_file = open(os.path.join(exp_dir, 'train.log'), 'w')

    
    # 记录系统真实的开始时间
    start_time_raw = time.time()
    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    init_msg = f" 实验开始，日志和权重保存在: {exp_dir}\n 训练开始时间: {start_time_str}\n"
    print(init_msg, end='')
    log_file.write(init_msg) # 把启动信息写入 log
    log_file.flush()

    train_dataset = RoadDataset(config['dataset']['root_path'], config['dataset']['name'], mode='train', img_size=config['dataset'].get('input_size', 1024))
    val_dataset   = RoadDataset(config['dataset']['root_path'], config['dataset']['name'], mode='val', img_size=config['dataset'].get('input_size', 1024))
    
    n_workers = config['dataset'].get('num_workers', 4)
    train_loader = DataLoader(train_dataset, batch_size=config['dataset']['batch_size'], shuffle=True, num_workers=n_workers, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=config['dataset']['batch_size'], shuffle=False, num_workers=n_workers, pin_memory=True)

    # 🌟先读取 img_size，再将其传递给 get_model，确保模型能拿到尺寸信息
    img_size = config['dataset'].get('input_size', 1024)
    model = get_model(config['model'], img_size=img_size).cuda()
    
    
    params, flops, fps = evaluate_model_complexity(model, device='cuda', img_size=img_size)
    
    # 制作排版字符串，并同时打印到屏幕和写入 log_file
    complexity_msg = (
        f"--------------------------------------------------\n"
        f" 📊 模型复杂度 @ 输入尺寸: {img_size}x{img_size}\n"
        f"    - 参数量 (Params):    {params / 1e6:.2f} M\n"
        f"    - 浮点运算量 (FLOPs): {flops / 1e9:.2f} G\n"
        f"    - 推理速度 (FPS):     {fps:.2f} 张/秒\n"
        f"--------------------------------------------------\n"
    )
    print(complexity_msg, end='')
    log_file.write(complexity_msg)
    log_file.flush() # 强制写入硬盘
    
    # === 从配置文件中提取超参数，如果没有配置则使用后面的默认值 ===
    weight_decay = config['training'].get('weight_decay', 1e-2)           # 权重衰减
    lr_factor = config['training'].get('lr_factor', 0.2)                  # 学习率衰减因子
    lr_patience = config['training'].get('lr_patience', 5)                # 学习率衰减的耐心值
    early_stop_patience = config['training'].get('early_stop_patience', 15) # 早停的耐心值

    optimizer = optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=weight_decay)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=lr_factor, patience=lr_patience)
    criterion = BCEDiceLoss().cuda()

    #  实例化 AMP 的梯度缩放器
    scaler = torch.amp.GradScaler('cuda')

    start_epoch = 1
    best_val_iou = 0.0 # 🌟 改为记录最高 IoU，初始值为 0
    epochs_without_improvement = 0
    evaluator = Evaluator(num_class=2) # 🌟 实例化评价器
    
    # 将超参数记录打印到日志，方便以后排查实验设置
    hyper_msg = (
        f" 训练超参数设置:\n"
        f"    - 权重衰减 (Weight Decay): {weight_decay}\n"
        f"    - 学习率调度 (LR Scheduler): ReduceLROnPlateau (factor={lr_factor}, patience={lr_patience})\n"
        f"    - 早停机制 (Early Stopping Patience): {early_stop_patience} 轮\n"
        f"--------------------------------------------------\n"
    )
    print(hyper_msg, end='')
    log_file.write(hyper_msg)
    log_file.flush()

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"🔄 发现断点文件，正在恢复: {args.resume}")
            checkpoint = torch.load(args.resume)
            
            start_epoch = checkpoint['epoch'] + 1
            best_val_iou = checkpoint.get('best_val_iou', 0.0) # 🌟 兼容读取 IoU
            epochs_without_improvement = checkpoint['epochs_without_improvement']
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            # 恢复 scaler 的状态
            if 'scaler_state_dict' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
            print(f"成功恢复！将从第 {start_epoch} 轮继续训练...")
            log_file.write(f"Resumed from {args.resume} at epoch {start_epoch}\n")
        else:
            print(f" 找不到断点文件: {args.resume}，将从头开始训练！")

    for epoch in range(start_epoch, config['training']['epochs'] + 1):
        # --- [A] 训练阶段 ---
        model.train()
        train_loss = 0.0

        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch [{epoch}/{config['training']['epochs']}] Train", leave=False)
        #必须在 train_loader_tqdm 里循环
        for batch_data in train_loader_tqdm:
            # 兼容 Dataset 返回 2 个或 3 个变量的情况
            if len(batch_data) == 3:
                imgs, masks, _ = batch_data  # 训练时不关心名字，用 _ 扔掉
            else:
                imgs, masks = batch_data
                
            imgs, masks = imgs.cuda(), masks.cuda()
            optimizer.zero_grad()
            
            # 加入 autocast 混合精度前向传播
            with torch.amp.autocast('cuda'):
                preds = model(imgs)
                
                # 🌟 核心修改 1：动态处理多输出模型
                if isinstance(preds, (tuple, list)):
                    # 如果是 SAM2-UNet (深度监督)，把三个尺度的 Loss 都算出来并相加
                    loss = sum([criterion(p, masks) for p in preds])
                else:
                    # 如果是普通模型 (比如原来的 AFDANet)，只有1个输出，正常算
                    loss = criterion(preds, masks)
            
            #  AMP 专用的反向传播和优化步骤
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            train_loader_tqdm.set_postfix({'loss': f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- [B] 验证阶段 ---
        model.eval()
        val_loss = 0.0
        evaluator.reset()
        with torch.no_grad():
            val_loader_tqdm = tqdm(val_loader, desc=f"Epoch [{epoch}/{config['training']['epochs']}] Valid", leave=False)
            #  在 val_loader_tqdm 里循环
            for batch_data in val_loader_tqdm:
                if len(batch_data) == 3:
                    imgs, masks, _ = batch_data
                else:
                    imgs, masks = batch_data
                    
                imgs, masks = imgs.cuda(), masks.cuda()
                
                with torch.amp.autocast('cuda'):
                    preds = model(imgs)
                    # 兼容 SAM2MS 等多输出模型
                    if isinstance(preds, (tuple, list)):
                        preds = preds[0]
                    loss = criterion(preds, masks) 
                
                val_loss += loss.item()
                
                # 🌟 将预测 Logits 转为概率，再二值化为 0 和 1 的 Numpy 数组
                preds_bin = (torch.sigmoid(preds) > 0.5).cpu().numpy().astype(int)
                masks_int = masks.cpu().numpy().astype(int)
                evaluator.add_batch(masks_int, preds_bin) # 调用你的 add_batch

                val_loader_tqdm.set_postfix({'loss': f"{loss.item():.4f}"})
                
        avg_val_loss = val_loss / len(val_loader)
        
        # 🌟 严格按照你 metrics.py 的逻辑提取指标
        _ = evaluator.Pixel_Precision() # 必须先算这两个
        _ = evaluator.Pixel_Recall()
        val_f1 = evaluator.Pixel_F1()
        val_iou = evaluator.Intersection_over_Union() # 提取道路类的 IoU
        
        # 打印并写入日志
        log_msg = f"Epoch [{epoch}/{config['training']['epochs']}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val IoU: {val_iou:.4f} | Val F1: {val_f1:.4f}\n"
        print(log_msg, end='')
        log_file.write(log_msg)
        log_file.flush()

        # --- [C] 保存现场与调度 ---
        scheduler.step(avg_val_loss)

        # 🌟 核心：按 IoU 是否破纪录来判断
        is_best = val_iou > best_val_iou
        if is_best:
            best_val_iou = val_iou
            epochs_without_improvement = 0 
        else:
            epochs_without_improvement += 1

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(), 
            'best_val_iou': best_val_iou, # 🌟 保存最优 IoU
            'epochs_without_improvement': epochs_without_improvement
        }
        
        torch.save(checkpoint, os.path.join(exp_dir, 'latest_model.pth'))

        if is_best:
            torch.save(checkpoint, os.path.join(exp_dir, 'best_model.pth'))
            msg = f"       🎉 验证集 IoU 创新高 ({best_val_iou:.4f})，已保存最佳权重！\n"
            print(msg, end='')
            log_file.write(msg)
        else:
            msg = f"     ⚠️ 连续 {epochs_without_improvement} 轮没有提升了。\n"
            print(msg, end='')
            log_file.write(msg)

        if epochs_without_improvement >= early_stop_patience:
            msg = f"🚫 连续 {early_stop_patience} 轮 IoU 未提升，触发早停！\n"
            print(msg, end='')
            log_file.write(msg)
            break

        log_file.flush()

    # --- [D] 训练彻底结束后的时间统计 ---
    end_time_raw = time.time()
    end_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # 计算总耗时 (秒转换为小时、分钟、秒)
    duration_sec = end_time_raw - start_time_raw
    hours, rem = divmod(duration_sec, 3600)
    minutes, seconds = divmod(rem, 60)
    duration_str = f"{int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒"

    # 生成最终的总结报告
    end_msg = (
        f"--------------------------------------------------\n"
        f" 训练结束时间: {end_time_str}\n"
        f" 整个实验总耗时: {duration_str}\n"
        f"🎉 实验完成！前往 {exp_dir} 查看结果。\n"
        f"--------------------------------------------------\n"
    )
    
    print(end_msg, end='')
    log_file.write(end_msg)
    log_file.close()

if __name__ == '__main__':
    main()