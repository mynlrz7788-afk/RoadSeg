

# 1. 导入搬运过来的模型
from .baselines.deeplabv3plus import DeepLabV3Plus
from .baselines.dinknet import DLinkNet34
from .baselines.Unet import UNet
from .baselines.AFDANet import AFDANet
from .custom.AFDANet_Exp1 import AFDANet_Exp1  # 🌟 导入实验1模型
from .custom.AFDANet_Exp2 import AFDANet_Exp2  # 🌟 导入实验1模型
from .custom.AFDANet_Exp3 import AFDANet_Exp3  # 🌟 导入实验1模型
from .custom.AFDANet_cov1 import AFDANet_cov1
from .custom.DSUNet import DSUNet
# 如果在 baselines 里加了 unet.py，就在这里 from .baselines.unet import UNet
# 导入魔改模型 (假设叫 AFDANet)
# from .custom.afdanet import AFDANet

def get_model(config):
    """模型工厂：根据配置文件的名字，自动返回对应的模型"""
    model_name = config['name']
    num_classes = config.get('num_classes', 1)

    if model_name == 'DeepLabV3Plus':
        return DeepLabV3Plus(n_classes=num_classes)
    elif model_name == 'DLinkNet':
        return DLinkNet34(num_classes=num_classes)
    elif model_name == 'UNet':
        return UNet()
    elif model_name == 'DSUNet':
        return DSUNet(n_classes=num_classes)
    
    # elif model_name == 'UNet':
    #     return UNet()
    # elif model_name == 'AFDANet':
    #     return AFDANet()
    # --- 如果以后魔改了 UNet 加了注意力机制 ---
    # elif model_name == 'UNet_Attention':
    #     return UNet_Attention()
    elif model_name == 'AFDANet':
        # 注意：AFDANet 初始化有 img_size 参数，默认 1024。
        # 如果你后续实验换了其他尺寸，可以在 config 里传过来，这里先用默认的
        return AFDANet(num_classes=num_classes)
    

    elif model_name == 'AFDANet_Exp1':
        return AFDANet_Exp1(num_classes=num_classes)
    elif model_name == 'AFDANet_Exp2':
        return AFDANet_Exp2(num_classes=num_classes)
    elif model_name == 'AFDANet_Exp3':
        return AFDANet_Exp3(num_classes=num_classes)
    elif model_name == 'AFDANet_cov1':
        return AFDANet_cov1(num_classes=num_classes)
    else:
        raise ValueError(f"没找到模型: {model_name}，请检查名字！")