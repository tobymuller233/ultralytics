from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_num_params
from ultralytics.nn.modules import Detect
import torch

model = YOLO("runs/yoloface500k_neumeta_baseline2/weights/best.pt")
for x in model.model.model:
    if isinstance(x, Detect):
        num_params = get_num_params(x)

total_params = get_num_params(model.model.model)
print(f"Detect: {num_params} parameters")
print(f"Other: {total_params - num_params} parameters")
print(f"All: {total_params} parameters")
