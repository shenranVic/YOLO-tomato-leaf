# -*- coding: utf-8 -*-

import warnings
import torch
import sys
sys.dont_write_bytecode = True # 禁止生成 .pyc

warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model='./ultralytics/cfg/models/11/yolo11_new.yaml')
    # model = YOLO(model='ultralytics/cfg/models/v8/yolov8.yaml')
    # model = YOLO(model='./runs/train/exp/weights/last.pt')
    # model.load('./runs/train-2025.8.31+ pure yolo+modifyParameters/exp/weights/best.pt') # 加载预训练权重,改进或者做对比实验时候不建议打开，因为用预训练模型整体精度没有很明显的提升
    print(model.model)
    model.train(data='./data.yaml',#数据集配置文件路径
                imgsz=640,
                epochs=100,
                batch=8,
                workers=0,#数据加载的工作线程数，显存爆设为0，默认8
                device='0',#用哪个显卡训练，空表示自动选择可用的GPU或CPU
                optimizer='SGD',#优化器类型
                close_mosaic=10,#在多少个epoch后关闭mosaic数据增强
                resume=True,#是否从上次中断的训练状态继续训练，False表示从头开始，True会继续训练
                project='./runs/train',#项目文件夹，保存训练结果
                name='exp',#命名保存的结果文件夹
                single_cls=False,#是否将所有类别视为一个类别，False表示保留原有类别
                cache=False,#是否缓存数据，设为False表示不缓存
                )
    # 将模型切换到部署模式（进行结构重参数化）
    for m in model.model.modules():
        if hasattr(m, 'switch_to_deploy'):
            m.switch_to_deploy()

    # 保存转换后的模型
    torch.save(model.state_dict(), 'yolo11_tomato.pt')