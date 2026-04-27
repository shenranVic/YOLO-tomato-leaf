本实验使用的数据集：[Tomato leaf diseases dataset for Object Detection](https://www.kaggle.com/datasets/sebastianpalaciob/tomato-leaf-diseases-dataset-for-object-detection)

### 消融实验结果对比

| Model | Atten | BiFPN | RepC3 | P | R | mAP@0.5 | F1 | Params (M) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| YOLOv11 | — | — | — | 0.88 | 0.88 | 0.934 | 0.88 | 2.46 |
| Atten | √ | — | — | 0.92 | 0.84 | 0.935 | 0.88 | 2.47 |
| BiFPN with RepC3 | — | √ | √ | 0.90 | 0.85 | 0.929 | 0.87 | 2.33 |
| BiFPN without RepC3 | — | √ | — | 0.88 | 0.86 | 0.931 | 0.87 | 1.98 |
| Atten+BiFPN without RepC3 | √ | √ | — | 0.90 | 0.83 | 0.927 | 0.86 | 1.98 |
| **Ours** | √ | √ | √ | 0.89 | 0.86 | 0.932 | 0.87 | 2.33 |

yaml文件路径：./ultralytics/cfg/models/11/yolo11_new.yaml

修改后的架构图：
<img width="655" height="571" alt="image" src="https://github.com/user-attachments/assets/87eeef4e-3e35-4fb4-a493-c5e64f379223" />



