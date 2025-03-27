import os
from ultralytics import YOLO
from pathlib import Path
import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory

pretrained = "weights/neumeta_model_v2_0227_dict.pt"
data_yaml = "ultralytics/cfg/datasets/SCUT_HEAD_A_B_stu_three_v1.yaml"
model_yaml = "ultralytics/cfg/models/yoloface500k/yoloface-500kp-layer21-dim120-3class.yaml"
# path_finetune = os.path.join(ROOT, "runs/yoloface500k_neumeta_baseline")
path_finetune = os.path.join(ROOT, "runs/debug")
model = YOLO(model_yaml)

model.train(data=data_yaml, epochs=700, batch=2, imgsz=640, workers=16, name=path_finetune, device="1", )
