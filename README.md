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

yaml文件如下：
nc: 9  # number of classes
scales:
  n: [0.50, 0.25, 1024]
#===========================Atten+BiFPN with RepC3（三合一）    开始===================
#backbone:
#  # [from, repeats, module, args]
#  - [-1, 1, Conv, [64, 3, 2]]                          # 0-P1/2
#  - [-1, 1, Conv, [128, 3, 2]]                         # 1-P2/4
#  - [-1, 2, C3k2, [256, False, 0.25]]                  # 2
#  - [-1, 1, SimAM, [256]]                              # 3  ← SimAM after first C3k2
#  - [-1, 1, Conv, [256, 3, 2]]                         # 4-P3/8
#  - [-1, 2, C3k2, [512, False, 0.25]]                  # 5
#  - [-1, 1, SimAM, [512]]                              # 6  ← SimAM after second C3k2
#  - [-1, 1, Conv, [512, 3, 2]]                         # 7-P4/16
#  - [-1, 2, C3k2, [512, True]]                         # 8
#  - [-1, 1, SimAM, [512]]                              # 9  ← SimAM after third C3k2
#  - [-1, 1, Conv, [1024, 3, 2]]                        # 10-P5/32
#  - [-1, 2, C3k2, [1024, True]]                        # 11
#  - [-1, 1, SimAM, [1024]]                             # 12 ← SimAM after fourth C3k2
#  - [-1, 1, SPPF, [1024, 5]]                           # 13
#  - [-1, 2, C2PSA, [1024]]                             # 14
#
#head:
#  # 层 15: BiFPN_RepC3。它在 task.py 里被拆分成了三个连续的层。
#  # 占用索引：15 (P3输出), 16 (P4输出), 17 (P5输出)
#  - [[3, 9, 14], 1, BiFPN_RepC3, [[256, 512, 1024]]]
#
#  # 层 18: Detect。接收 15, 16, 17 层的输出
#  - [[15, 16, 17], 1, Detect, [nc]]
#===========================Atten+BiFPN with RepC3（三合一）   结束===================



#===========================Atten    开始===================
#backbone:
#  # [from, repeats, module, args]
#  - [-1, 1, Conv, [64, 3, 2]]                          # 0-P1/2
#  - [-1, 1, Conv, [128, 3, 2]]                         # 1-P2/4
#  - [-1, 2, C3k2, [256, False, 0.25]]                  # 2
#  - [-1, 1, SimAM, [256]]                              # 3  ← SimAM after first C3k2
#  - [-1, 1, Conv, [256, 3, 2]]                         # 4-P3/8
#  - [-1, 2, C3k2, [512, False, 0.25]]                  # 5
#  - [-1, 1, SimAM, [512]]                              # 6  ← SimAM after second C3k2
#  - [-1, 1, Conv, [512, 3, 2]]                         # 7-P4/16
#  - [-1, 2, C3k2, [512, True]]                         # 8
#  - [-1, 1, SimAM, [512]]                              # 9  ← SimAM after third C3k2
#  - [-1, 1, Conv, [1024, 3, 2]]                        # 10-P5/32
#  - [-1, 2, C3k2, [1024, True]]                        # 11
#  - [-1, 1, SimAM, [1024]]                             # 12 ← SimAM after fourth C3k2
#  - [-1, 1, SPPF, [1024, 5]]                           # 13
#  - [-1, 2, C2PSA, [1024]]                             # 14
#
#head:
#  # --- 第一阶段: Top-Down (P5 -> P4 -> P3) ---
#  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 15 (P5 up)
#  - [[-1, 9], 1, Concat, [1]] # 16 cat backbone P4 (索引 9 是第三个 SimAM)
#  - [-1, 2, C3k2, [512, False]] # 17
#
#  - [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 18 (P4 up)
#  - [[-1, 6], 1, Concat, [1]] # 19 cat backbone P3 (索引 6 是第二个 SimAM)
#  - [-1, 2, C3k2, [256, False]] # 20 (P3/8-small)
#
#  # --- 第二阶段: Bottom-Up (P3 -> P4 -> P5) ---
#  - [-1, 1, Conv, [256, 3, 2]] # 21
#  - [[-1, 17], 1, Concat, [1]] # 22 cat head P4 (索引 17 是上面生成的层)
#  - [-1, 2, C3k2, [512, False]] # 23 (P4/16-medium)
#
#  - [-1, 1, Conv, [512, 3, 2]] # 24
#  - [[-1, 14], 1, Concat, [1]] # 25 cat backbone P5 (索引 14 是 C2PSA 后的输出)
#  - [-1, 2, C3k2, [1024, True]] # 26 (P5/32-large)
#
#  # --- 检测头 ---
#  - [[20, 23, 26], 1, Detect, [nc]] # Detect(P3, P4, P5)
#===========================Atten   结束===================


#===========================BiFPN without RepC3 +Atten  开始===================
#backbone:
#  # [from, repeats, module, args]
#  - [-1, 1, Conv, [64, 3, 2]]                          # 0-P1/2
#  - [-1, 1, Conv, [128, 3, 2]]                         # 1-P2/4
#  - [-1, 2, C3k2, [256, False, 0.25]]                  # 2
#  - [-1, 1, SimAM, [256]]                              # 3  ← SimAM after first C3k2
#  - [-1, 1, Conv, [256, 3, 2]]                         # 4-P3/8
#  - [-1, 2, C3k2, [512, False, 0.25]]                  # 5
#  - [-1, 1, SimAM, [512]]                              # 6  ← SimAM after second C3k2
#  - [-1, 1, Conv, [512, 3, 2]]                         # 7-P4/16
#  - [-1, 2, C3k2, [512, True]]                         # 8
#  - [-1, 1, SimAM, [512]]                              # 9  ← SimAM after third C3k2
#  - [-1, 1, Conv, [1024, 3, 2]]                        # 10-P5/32
#  - [-1, 2, C3k2, [1024, True]]                        # 11
#  - [-1, 1, SimAM, [1024]]                             # 12 ← SimAM after fourth C3k2
#  - [-1, 1, SPPF, [1024, 5]]                           # 13
#  - [-1, 2, C2PSA, [1024]]                             # 14
#
#head:
#  # 层 15: BiFPN_RepC3。它在 task.py 里被拆分成了三个连续的层。
#  # 占用索引：15 (P3输出), 16 (P4输出), 17 (P5输出)
#  - [[3, 9, 14], 1, BiFPN_Ablation, [[256, 512, 1024]]]
#
#  # 层 18: Detect。接收 15, 16, 17 层的输出
#  - [[15, 16, 17], 1, Detect, [nc]]
#===========================BiFPN without RepC3 +Atten  结束===================


#===========================BiFPN without RepC3  开始===================
#backbone:
#  # [from, repeats, module, args]
#  - [-1, 1, Conv, [64, 3, 2]]                          # 0-P1/2
#  - [-1, 1, Conv, [128, 3, 2]]                         # 1-P2/4
#  - [-1, 2, C3k2, [256, False, 0.25]]                  # 2-P3_Raw (原索引 2)
#  # 去掉原索引 3 的 SimAM
#  - [-1, 1, Conv, [256, 3, 2]]                         # 3-P3/8 (原索引 4)
#  - [-1, 2, C3k2, [512, False, 0.25]]                  # 4-P4_Raw (原索引 5)
#  # 去掉原索引 6 的 SimAM
#  - [-1, 1, Conv, [512, 3, 2]]                         # 5-P4/16 (原索引 7)
#  - [-1, 2, C3k2, [512, True]]                         # 6 (原索引 8)
#  # 去掉原索引 9 的 SimAM
#  - [-1, 1, Conv, [1024, 3, 2]]                        # 7-P5/32 (原索引 10)
#  - [-1, 2, C3k2, [1024, True]]                        # 8 (原索引 11)
#  # 去掉原索引 12 的 SimAM
#  - [-1, 1, SPPF, [1024, 5]]                           # 9 (原索引 13)
#  - [-1, 2, C2PSA, [1024]]                             # 10 (原索引 14)
#
#head:
#  # 这里的引用索引必须根据上面新的索引进行修改：
#  # 原 [3, 9, 14] 对应现在的 [2, 6, 10]
#  - [[2, 6, 10], 1, BiFPN_Ablation, [[256, 512, 1024]]] # 11 (占用 11, 12, 13)
#
#  # Detect 层引用 BiFPN 的三个输出
#  - [[11, 12, 13], 1, Detect, [nc]]                     # 14
#===========================BiFPN without RepC3  结束===================



#===========================BiFPN with RepC3  开始===================
backbone:
  # [from, repeats, module, args]
  - [-1, 1, Conv, [64, 3, 2]]                          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]                         # 1-P2/4
  - [-1, 2, C3k2, [256, False, 0.25]]                  # 2-P3_Raw (原索引 2)
  # 去掉原索引 3 的 SimAM
  - [-1, 1, Conv, [256, 3, 2]]                         # 3-P3/8 (原索引 4)
  - [-1, 2, C3k2, [512, False, 0.25]]                  # 4-P4_Raw (原索引 5)
  # 去掉原索引 6 的 SimAM
  - [-1, 1, Conv, [512, 3, 2]]                         # 5-P4/16 (原索引 7)
  - [-1, 2, C3k2, [512, True]]                         # 6 (原索引 8)
  # 去掉原索引 9 的 SimAM
  - [-1, 1, Conv, [1024, 3, 2]]                        # 7-P5/32 (原索引 10)
  - [-1, 2, C3k2, [1024, True]]                        # 8 (原索引 11)
  # 去掉原索引 12 的 SimAM
  - [-1, 1, SPPF, [1024, 5]]                           # 9 (原索引 13)
  - [-1, 2, C2PSA, [1024]]                             # 10 (原索引 14)

head:
  # 这里的引用索引必须根据上面新的索引进行修改：
  # 原 [3, 9, 14] 对应现在的 [2, 6, 10]
  - [[2, 6, 10], 1, BiFPN_RepC3, [[256, 512, 1024]]] # 11 (占用 11, 12, 13)

  # Detect 层引用 BiFPN 的三个输出
  - [[11, 12, 13], 1, Detect, [nc]]                     # 14
#===========================BiFPN with RepC3  结束===================
