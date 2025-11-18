# src/explainability.py
import cv2
from ultralytics import YOLO
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
import torchvision.transforms as T

# NOTE: this script assumes a PyTorch backbone available in the YOLO model.
# For simplicity we will get a detection, crop bbox, run GradCAM on the model's backbone (if accessible).

MODEL_WEIGHTS = 'runs/detect/defectvision_exp/weights/best.pt'
SOURCE_IMAGE = 'assets/sample.jpg'  # replace with a path to test image

def get_backbone_and_target(model):
    # Attempt to access backbone (this might vary by ultralytics version)
    # We'll try common attributes. If not found, this script will need small edits.
    try:
        backbone = model.model.model[0]  # may vary
    except Exception:
        backbone = None
    return backbone

def main():
    model = YOLO(MODEL_WEIGHTS)
    res = model(SOURCE_IMAGE)[0]
    img_bgr = cv2.imread(SOURCE_IMAGE)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32)/255.0

    if len(res.boxes) == 0:
        print("No detections found.")
        return

    # pick the first detection for demo
    box = res.boxes.xyxy[0].cpu().numpy().astype(int)  # x1,y1,x2,y2
    conf = float(res.boxes.conf[0].cpu().numpy())
    cls = int(res.boxes.cls[0].cpu().numpy())

    crop = img_bgr[box[1]:box[3], box[0]:box[2]]
    if crop.size == 0:
        print("Invalid crop, skip.")
        return

    # prepare crop for model
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224,224)),
        T.ToTensor(),
    ])
    input_tensor = transform(crop).unsqueeze(0)

    # Try to access a backbone (highly dependent on ultralytics internal model structure)
    try:
        backbone = model.model.model[0]  # may require change per ultralytics version
        # use GradCAM on backbone last conv layer - user may need to adjust target_layer
        target_layer = None
        # find a Conv2d layer: search backwards
        for m in reversed(list(backbone.modules())):
            if isinstance(m, torch.nn.Conv2d):
                target_layer = m
                break
        if target_layer is None:
            print("Could not find Conv2d layer in backbone for GradCAM.")
            return

        cam = GradCAM(model=backbone, target_layers=[target_layer], use_cuda=False)
        targets = [ClassifierOutputTarget(cls)]
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
        rgb_img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB).astype(np.float32)/255.0
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Grad-CAM on crop", cam_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        print("Grad-CAM failed (structure mismatch). Error:", e)
        print("You can still visualize detections with detect.py")

if __name__ == '__main__':
    main()
