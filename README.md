# 🍊 Orange Fruit Quality Assessment (YOLOv11)

This project detects and classifies orange fruit quality using YOLOv8 and a custom dataset.

---

## 1. 🧪 Install Anaconda

- Download: [Anaconda Download](https://anaconda.com/download)
- During install: click "Skip Registration", use default options.
- Open **Anaconda Prompt** from the Start menu.

---

## 2. ⚙️ Create YOLO Environment

```bash
conda create --name yolo11-env python=3.12 -y
conda activate yolo11-env
pip install ultralytics
# For GPU (CUDA 12.4):
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Verify GPU:**
```bash
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

---

## 3. 🖼 Gather & Label Dataset

- Collect 100+ images of oranges.
- Label using [LabelImg](https://github.com/tzutalin/labelImg) or [Label Studio](https://labelstud.io/).
- Save labels in YOLO TXT format.

**Folder structure:**
```
my_dataset/
├── images/
│   ├── image1.jpg
│   └── ...
├── labels/
│   ├── image1.txt
│   └── ...
└── classes.txt
```
**Example Dataset:**  
- [Fresh/Rotten Orange Classifier on Roboflow](https://universe.roboflow.com/neha-chandekar-yxsnl/fresh-rotten-orange-classifier/)

---

## 4. 📁 Setup Folder Structure

Open a terminal in your project's root directory and run:

```bash
mkdir yolo
cd yolo
mkdir data
```

**Download train/val split script:**
```bash
curl --output yolo_train_val_split.py https://raw.githubusercontent.com/EdjeElectronics/Train-and-Deploy-YOLO-Models/refs/heads/main/utils/train_val_split.py
```

**Run the script:**
```bash
python yolo_train_val_split.py --datapath="my_dataset" --train_pct=.8
```

**Result:**
```
yolo/data/
├── train/
│   ├── images/
│   └── labels/
└── validation/
    ├── images/
    └── labels/
```

---

## 5. 📝 Create `data.yaml`

Create `data.yaml` in your `yolo` folder:

```yaml
path: ./data
train: train/images
val: validation/images

nc: 2
names: ["Fresh Oranges", "Rotten Oranges"]
```

---

## 6. 🏋️ Train the Model

```bash
yolo detect train data=data.yaml model=yolo11s.pt epochs=40 imgsz=480
```

- Model options:  
  - `yolo11n.pt` → nano (fastest, lowest accuracy)  
  - `yolo11s.pt` → small (balanced)  
  - `yolo11m.pt` → medium  
  - `yolo11l.pt` → large (highest accuracy, slowest)

**Output:**  
Model weights saved to:  
```
runs/detect/train/weights/best.pt
```

---

## 7. 🚀 Run Real-Time Detection

Use the provided `app.py` to run real-time detection with your webcam:

```bash
streamlit run app.py
```

---

## 📂 Repo Structure

```
Orange Fruit Quality Assesment/
├── app.py
├── yolo/
│   ├── data/
│   ├── runs/
│   ├── yolo_train_val_split.py
│   ├── yolo11n.pt
│   └── yolo11s.pt
└── ...
```

---

## 📌 References

- [Fresh/Rotten Orange Classifier Dataset (Roboflow)](https://universe.roboflow.com/neha-chandekar-yxsnl/fresh-rotten-orange-classifier/)
- [YouTube: Orange Quality Detection Demo](https://www.youtube.com/watch?v=r0RspiLG260)
- [Train YOLO Models - EJTech Guide](https://www.ejtech.io/learn/train-yolo-models)
- [Ultralytics YOLO Docs](https://docs.ultralytics.com/)
- [LabelImg](https://github.com/tzutalin/labelImg)
- [Label Studio](https://labelstud.io/)

---
