from ultralytics import YOLO
import argparse
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory

parser = argparse.ArgumentParser()
parser.add_argument("--weight", "-w", type=str, default="runs/yoloface500k_neumeta_baseline2/weights/best.pt")
parser.add_argument("--format", "-f", type=str, default="onnx")
parser.add_argument("--opset", "-o", type=int, default=15, help="onnx opset version")
args = parser.parse_args()

pretrained = args.weight
model = YOLO(pretrained)
model.export(format=args.format, opset=args.opset)

