# Automated Visual Pollution Detection from Street Imagery

This repository presents an AI-based system for detecting visual pollutants from urban street imagery using object detection techniques. The solution was designed to support city planners and municipalities in evaluating and monitoring environmental cleanliness.

> 📝 This work has been academically published in the *International Journal on Recent and Innovation Trends in Computing and Communication (IJRITCC)*:  
> [Read the Paper](https://ijritcc.org/index.php/ijritcc/article/view/7030)

---

## 📌 Overview

The goal of this project is to classify and localize various urban visual pollutants captured via street-level images taken from moving vehicles. These include:

- Graffiti
- Potholes
- Litter
- Broken signage
- Faded billboards
- Dim streetlights
- Cluttered sidewalks

The model uses **YOLOv5** for object detection and is trained on a manually annotated dataset with real-world imagery.

---

## 📊 Visual Pollution Index (VPI)

In addition to detection, this project introduces a novel **Visual Pollution Index (VPI)** which numerically scores street images based on the severity and frequency of pollutants detected.

This index can serve as a smart city metric to prioritize municipal cleanup operations.

---

## 🚀 Running the Project

### 1. Clone the repo
```bash
git clone https://github.com/UmeshPatekar23/visual-pollution-detector.git
cd visual-pollution-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the model
```bash
python inference.py --source dataset/test --weights models/best.pt
```

### 4. Generate VPI scores
```bash
python scripts/vpi_calculator.py --input results/predictions.csv
```

---

## 📁 Project Structure

```
├── dataset/
├── models/
├── results/
├── scripts/
├── paper/
├── README.md
├── requirements.txt
└── inference.py
```

---
