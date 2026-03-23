import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math

class DynamicScaleConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super(DynamicScaleConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = nn.Parameter(torch.zeros(out_channels))

        self.scale_predictor = nn.Conv2d(in_channels, 2, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.register_buffer('kernel_coords', self._create_kernel_coords(kernel_size))

        nn.init.kaiming_normal_(self.weight, mode='fan_out', nonlinearity='relu')

    def _create_kernel_coords(self, kernel_size):
        center = kernel_size // 2
        y, x = torch.meshgrid(
            torch.linspace(-center, center, kernel_size),
            torch.linspace(-center, center, kernel_size),
            indexing='ij'
        )
        coords = torch.stack([x, y], dim=-1).view(-1, 2)
        return coords

    def forward(self, x):
        pass # 可视化时不直接调用前向传播

def visualize_interpolation_and_aggregation(model, input_tensor, pixel_coords=(16, 16)):
    model.eval()
    with torch.no_grad():
        scale_params = model.scale_predictor(input_tensor)
        
        sx = torch.exp(scale_params[0, 0, pixel_coords[0], pixel_coords[1]]).item()
        sy = torch.exp(scale_params[0, 1, pixel_coords[0], pixel_coords[1]]).item()
        
        if abs(sx - 1.0) < 0.2 and abs(sy - 1.0) < 0.2:
            sx, sy = 1.8, 2.5 
            
        orig_coords = model.kernel_coords.cpu().numpy()
        deformed_coords = orig_coords.copy()
        deformed_coords[:, 0] *= sx
        deformed_coords[:, 1] *= sy
        
        center_x, center_y = pixel_coords[1], pixel_coords[0]
        orig_abs = orig_coords + np.array([center_x, center_y])
        def_abs = deformed_coords + np.array([center_x, center_y])
        
        plt.figure(figsize=(12, 12))
        
        min_x = math.floor(np.min(def_abs[:, 0])) - 1
        max_x = math.ceil(np.max(def_abs[:, 0])) + 1
        min_y = math.floor(np.min(def_abs[:, 1])) - 1
        max_y = math.ceil(np.max(def_abs[:, 1])) + 1
        
        plt.xticks(np.arange(min_x, max_x + 1, 1))
        plt.yticks(np.arange(min_y, max_y + 1, 1))
        plt.grid(True, linestyle='-', alpha=0.3, color='gray')
        
        # 1. 绿点: 双线性插值的数据源
        green_plotted = set()
        for i in range(len(def_abs)):
            bx, by = def_abs[i]
            x0, x1 = math.floor(bx), math.ceil(bx)
            y0, y1 = math.floor(by), math.ceil(by)
            
            neighbors = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
            for nx, ny in neighbors:
                if (nx, ny) not in green_plotted:
                    plt.scatter(nx, ny, c='green', marker='s', s=80, alpha=0.3, 
                                label='1. Source Pixels (Green)' if len(green_plotted)==0 else "")
                    green_plotted.add((nx, ny))
                plt.plot([bx, nx], [by, ny], color='green', linestyle=':', alpha=0.5, zorder=1)

        # 2. 红点: 原始卷积坐标
        plt.scatter(orig_abs[:, 0], orig_abs[:, 1], c='red', label='2. Original Grid (Red)', s=100, zorder=4)
        
        # 3. 蓝点: 拉伸后的采样坐标
        plt.scatter(def_abs[:, 0], def_abs[:, 1], c='blue', label='3. Deformed Sample Points (Blue)', s=150, zorder=5)

        # 4. 橙色星号: 最终特征的汇聚点 (输出像素)
        plt.scatter(center_x, center_y, c='orange', marker='*', edgecolors='black', s=600, 
                    label='4. Final Output Point (Orange Star)', zorder=10)

        for i in range(len(orig_abs)):
            # 灰线: 红点到蓝点的变形轨迹 (拉伸过程)
            plt.arrow(orig_abs[i, 0], orig_abs[i, 1], 
                      def_abs[i, 0] - orig_abs[i, 0], 
                      def_abs[i, 1] - orig_abs[i, 1], 
                      head_width=0.08, head_length=0.1, fc='gray', ec='gray', alpha=0.3, zorder=2)
            
            # 橙色点划线: 蓝点带着算好的特征值，汇聚到中心橙色星号的过程 (卷积加权求和)
            plt.plot([def_abs[i, 0], center_x], [def_abs[i, 1], center_y], color='orange', linestyle='-.', alpha=0.6, linewidth=2, zorder=3)
            
            plt.annotate(f"{i}", (def_abs[i, 0], def_abs[i, 1]), textcoords="offset points", xytext=(0,10), ha='center', color='blue', weight='bold')

        plt.title(f"Full Convolution Process: Deform -> Interpolate -> Aggregate\nOutput Pixel (X:{center_x}, Y:{center_y}) | sx={sx:.2f}, sy={sy:.2f}", fontsize=14)
        plt.legend(loc='upper right', fontsize=10)
        plt.axis('equal')
        plt.gca().invert_yaxis() 
        
        save_path = "final_aggregation_viz.png"
        plt.savefig(save_path, bbox_inches='tight')
        print(f"终极版可视化图像已保存至: {save_path}，请查看！")

if __name__ == "__main__":
    m = DynamicScaleConv2d(in_channels=3, out_channels=16)
    sample_input = torch.randn(1, 3, 32, 32) 
    visualize_interpolation_and_aggregation(m, sample_input, pixel_coords=(16, 16))