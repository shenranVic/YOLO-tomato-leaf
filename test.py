# -*- coding: utf-8 -*-
import warnings
import torch
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from thop import profile

# 加载修改后的模型
# model_repc3 = Model('path/to/your/modified_yaml.yaml')
# model_repc3  = YOLO(model='./ultralytics/cfg/models/11/yolo11.yaml')
# model1 = YOLO(model='runs/train-2025.9.23 yolo+SImAM+CA+BiFPN+repC3/exp/weights/best.pt')
# model_repc3 = YOLO(model='runs/train-2025.9.23 yolo+SImAM+CA+BiFPN+repC3/exp/weights/best.pt')
model = YOLO(model='runs/train/exp/weights/best.pt')
print(model)
# # 打印特定层的信息来验证
# print("替换后的模型结构：")
# for name, module in model_repc3.named_modules():
#     if 'RepC3' in name:
#         print(f"层: {name}, 模块: {module}")
# dummy_input = torch.randn(1, 3, 640, 640)
# # 计算新的FLOPs
# flops_new, params_new = profile(model_repc3, inputs=(dummy_input,))
# print(f"替换后模型 - FLOPs: {flops_new/1e9:.2f}G, Params: {params_new/1e6:.2f}M")

# # 比较变化
# print(f"FLOPs变化: {(flops_new-flops)/flops*100:.2f}%")
# print(f"Params变化: {(params_new-params)/params*100:.2f}%")