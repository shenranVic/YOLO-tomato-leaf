# -*- coding: utf-8 -*-


from ultralytics import YOLO
from thop import profile
import torch

if __name__ == '__main__':

    # 原YOLOv11模型
    # model = YOLO(model='runs/train-2025.8.30pure yolo/exp6/weights/best.pt')
    # #Yolo+注意力机制
    model = YOLO(model='runs/train/exp/weights/best.pt')
    # #YOLO+BiFPN+注意力机制
    # model = YOLO(model='runs/train-biFPN+CA+SimAM/exp/weights/best.pt')
    # #YOLO+BiFPN+注意力机制+RepC3
    # model1 = YOLO(model='runs/train-2025.9.23 yolo+SImAM+CA+BiFPN+repC3/exp/weights/best.pt')
    # model2 = YOLO(model='runs/train-2025.9.23 yolo+SImAM+CA+BiFPN+repC3/exp/weights/bestfused.pt')
    # model3 = YOLO(model='runs/train-2025.10.4yolo+simAM+CA+BiFPN+repC3_jianzhi/exp3/weights/best.pt')

    #Yolov5
    # model = YOLO(model='runs/yolov5m.pt')
    #Yolov8
    # model = YOLO(model='runs/yolov8n.pt')


    print("--------------------------------")
    metrics1 = model.val(data='data.yaml')
    print('P (mp)        :', metrics1.box.mp)  # Precision
    print('R (mr)        :', metrics1.box.mr)  # Recall
    print('mAP50         :', metrics1.box.map50)  # mAP@0.5
    print('mAP50-95      :', metrics1.box.map)  # mAP@0.5:0.95
    model.info()
    print("--------------------------------")

    # model.predict(source='data/test/images',
    #               #该参数可以填入需要推理的图片或者视频路径，如果打开摄像头推理则填入0就行
    #               save=True,#该参数填入True, 代表把推理结果保存下来， 默认是不保存的， 所以一般都填入True
    #               show=False,#该参数填入True，代表把推理结果以窗口形式显示出来， 默认是显示,不显示填False
    #               )
