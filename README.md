# DefectVision

**DefectVision** — Real-time surface defect detection (cracks, rust, dents) using YOLO + CNN backbones and OpenCV preprocessing. Designed for research-grade results and optimized for edge deployment.

## Key features
- YOLO-based object detection (YOLOv8 recommended) with EfficientNet/ResNet backbone
- OpenCV preprocessing and classical filters for contrast & edge enhancement
- Data augmentation pipeline for small-defect robustness
- Explainability via Grad-CAM and frequency-domain analysis
- Model compression: pruning, quantization (TFLite / ONNX)
- Inference script for webcam / video and batch images
- Evaluation: mAP, IoU, precision/recall, F1


### 📦 Dataset Setup
This project does not include the full dataset due to size.
Download the dataset from:

→ NEU Surface Defects Dataset  
→ DAGM Defects Dataset  
→ Or use your own dataset

Place your images and labels under:

data/images/train  
data/images/val  
data/labels/train  
data/labels/val


## Quick start

1. Create environment:
```bash
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
