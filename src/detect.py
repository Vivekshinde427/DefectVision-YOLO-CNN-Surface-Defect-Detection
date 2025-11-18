# Authoer: Vivek Shinde
import argparse
from ultralytics import YOLO
import cv2
import os

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='runs/detect/defectvision_exp/weights/best.pt')
    p.add_argument('--source', type=str, default='0', help='0 for webcam or path to image/video')
    p.add_argument('--save', action='store_true', help='save output annotated video/images')
    p.add_argument('--device', type=str, default='auto')
    return p.parse_args()

def main():
    args = parse_args()
    model = YOLO(args.weights)

    src = args.source
    is_cam = src.isdigit()
    cap = cv2.VideoCapture(int(src) if is_cam else src)

    out_writer = None
    if args.save and not is_cam:
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter('output_annotated.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)  # inference
        annotated = results[0].plot()
        cv2.imshow('DefectVision', annotated)
        if args.save and not is_cam:
            out_writer.write(annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
