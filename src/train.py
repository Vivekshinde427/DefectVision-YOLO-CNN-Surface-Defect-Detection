# src/train.py
import argparse
from ultralytics import YOLO
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='configs/data.yaml', help='path to data yaml')

    p.add_argument('--model', type=str, default='yolov8n.pt', help='base model (yolov8n.pt / yolov8s.pt...)')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch', type=int, default=8)
    p.add_argument('--name', type=str, default='defectvision_exp')

    p.add_argument('--device', type=str, default='auto', help='cuda or cpu or auto')

    return p.parse_args()

def main():
    args = parse_args()

    data_path = args.data
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}. Update --data to point to your data.yaml")

    print("Starting training with config:")
    print(vars(args))

    model = YOLO(args.model)  
    model.train(
        data=data_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        name=args.name,
        device=args.device
    )

    print("Training finished. Weights saved under runs/")

if __name__ == '__main__':
    main()
