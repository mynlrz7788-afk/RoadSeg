import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class DynamicScaleConv2d(nn.Module):
    """
    简洁的仿射自适应卷积
    对每个像素预测缩放参数，应用到3x3卷积核坐标上
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DynamicScaleConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # 基础卷积权重
        self.weight = nn.Parameter(
            torch.randn(out_channels, in_channels, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(out_channels))

        # 缩放参数预测 - 每个像素预测2个参数
        self.scale_predictor = nn.Conv2d(in_channels, 2, kernel_size=kernel_size, stride=stride, padding=padding,
                                         dilation=dilation)

        # 基础卷积核坐标 (3x3网格)
        self.register_buffer('kernel_coords', self._create_kernel_coords(kernel_size))
        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')


        # === 新增代码开始 ===
        # 将负责预测缩放比例的卷积层权重和偏置全部初始化为 0
        nn.init.constant_(self.scale_predictor.weight, 0)
        nn.init.constant_(self.scale_predictor.bias, 0)
        # === 新增代码结束 ===

    def _create_kernel_coords(self, kernel_size):
        """创建基础卷积核坐标"""
        center = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-center, center, kernel_size),
            torch.linspace(-center, center, kernel_size),
            indexing='ij'
        )
        coords = torch.stack([x, y], dim=-1).view(-1, 2)
        return coords  # [k*k, 2]

    def forward(self, x):

        # 预测缩放参数 [B, 2, H, W]
        scale_params = self.scale_predictor(x)

        # 提取参数
        sx = scale_params[:, 0, :, :]  # x轴缩放因子 [B, H, W]
        sy = scale_params[:, 1, :, :]  # y轴缩放因子 [B, H, W]

        # # 应用指数函数保证缩放因子为正
        # sx = torch.exp(sx)
        # sy = torch.exp(sy)

        # 🌟 关键修复：防止指数爆炸导致 float16 溢出
        # 将网络预测的系数限制在 [-2.0, 2.0] 之间，然后再进行 exp 指数运算
        # 这样网络一开始预测 0 时，exp(0) = 1.0 (等价于标准 3x3 卷积，完美收敛)
        # exp(2) ≈ 7.38 (最大拉伸7倍)，exp(-2) ≈ 0.13 (最大收缩到0.13倍)
        sx = torch.exp(torch.clamp(sx, min=-2.0, max=2.0))
        sy = torch.exp(torch.clamp(sy, min=-2.0, max=2.0))

        # 构建缩放矩阵 [B, 2, 2, H, W]
        # 缩放矩阵: [[sx, 0], [0, sy]]
        zero = torch.zeros_like(sx)

        # 重塑为每个像素的仿射矩阵 [B, H, W, 2, 2]
        scaling_matrices = torch.stack([
            torch.stack([sx, zero], dim=-1),  # 第一行: [sx, 0]
            torch.stack([zero, sy], dim=-1)  # 第二行: [0, sy]
        ], dim=-2)  # shape: [B, H, W, 2, 2]

        # 保存缩放矩阵用于可视化
        self.scaling_matrices = scaling_matrices.detach()

        # 应用缩放变换到卷积核坐标
        transformed_coords = self._transform_coordinates(scaling_matrices)  # [B, H, W, 9, 2]

        # 使用变换后的坐标进行采样
        output = self._apply_deformable_conv_optimized(x, transformed_coords)

        return output

    def _transform_coordinates(self, scaling_matrices):
        """应用缩放变换到卷积核坐标"""
        B, H, W, _, _ = scaling_matrices.shape

        # 扩展基础坐标 [1, 1, 1, k*k, 2] -> [B, H, W, k*k, 2]
        base_coords = self.kernel_coords.view(1, 1, 1, self.kernel_size ** 2, 2).expand(B, H, W, self.kernel_size ** 2, 2)
        transformed = torch.einsum('bhwki,bhwji->bhwkj', base_coords, scaling_matrices)

        return transformed

    def _apply_deformable_conv_optimized(self, x, transformed_coords):
        B, C, H, W = x.shape
        k = self.kernel_size
        k_sq = k * k

        # 预计算基础网格 [B, H, W, 2]
        base_grid = self._create_base_grid(B, H, W, x.device)

        # 重塑权重 [out_channels, in_channels, k, k] -> [out_channels, in_channels, k_sq]
        weight_flat = self.weight.view(self.out_channels, self.in_channels, k_sq)

        # 初始化输出
        output = torch.zeros(B, self.out_channels, H, W, device=x.device)

        # 只对卷积核位置循环，其余并行
        for k_idx in range(k_sq):
            # 获取当前核位置的偏移 [B, H, W, 2]
            offset = transformed_coords[:, :, :, k_idx, :]

            # 创建采样网格 [B, H, W, 2]
            sampling_grid = base_grid + offset
            sampling_grid = self._normalize_grid(sampling_grid, H, W)

            # 对所有输入通道同时采样 [B, C, H, W]
            sampled_all = F.grid_sample(
                x, sampling_grid,
                mode='bilinear', padding_mode='reflection', align_corners=True
            )

            # 对所有输出通道同时计算加权和
            # weight_flat[:, :, k_idx] 形状: [out_channels, in_channels]
            # sampled_all 形状: [B, in_channels, H, W]
            # 使用einsum进行批量矩阵乘法: [B, out_channels, H, W] += [out_channels, in_channels] @ [B, in_channels, H, W]
            output += torch.einsum('oi,bihw->bohw', weight_flat[:, :, k_idx], sampled_all)

        # 添加偏置
        output += self.bias.view(1, -1, 1, 1)

        return output

    def _create_base_grid(self, B, H, W, device):
        """创建基础位置网格"""
        y, x = torch.meshgrid(
            torch.arange(H, device=device).float(),
            torch.arange(W, device=device).float(),
            indexing='ij'
        )

        base_grid = torch.stack([x, y], dim=-1)  # [H, W, 2]
        base_grid = base_grid.unsqueeze(0).expand(B, H, W, 2)  # [B, H, W, 2]

        return base_grid

    def _normalize_grid(self, grid, H, W):
        """归一化网格到 [-1, 1]"""
        grid = grid.clone()
        grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (W - 1) - 1.0
        grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (H - 1) - 1.0
        return grid.clip(-1, 1)



if __name__ == '__main__':
    # 创建模型和输入
    model = DynamicScaleConv2d(in_channels=3, out_channels=16)
    x = torch.randn(1, 3, 32, 32)  # 随机输入

    # 前向传播（会保存中间变量）
    output = model(x)

    print(output.shape)