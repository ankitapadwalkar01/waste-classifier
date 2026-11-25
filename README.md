# WasteWise AI

### Lightweight CNN with Knowledge Distillation for Real-Time Waste Classification

WasteWise AI is a computer vision project that classifies waste
images into nine categories using a lightweight convolutional
neural network.

## Project Overview

The project explores efficient image classification using:

- ResNet-50 Teacher Network
- Lightweight Student CNN
- Depthwise Separable Convolutions
- Inverted Residual Blocks
- Squeeze-and-Excitation Attention
- Knowledge Distillation
- Grad-CAM Explainability

## Dataset

UCI RealWaste Dataset

Total Images: 4,752

Number of Classes: 9

Classes:
1. Cardboard
2. Food Organics
3. Glass
4. Metal
5. Miscellaneous Trash
6. Paper
7. Plastic
8. Textile Trash
9. Vegetation

The dataset is not included in this repository.
Please obtain it from the original dataset source.

## Test Results

| Model | Accuracy | Macro F1 |
|---|---:|---:|
| ResNet-50 Teacher | 89.90% | 90.34% |
| Student CE | 72.65% | 72.12% |
| Student KD | 71.53% | 71.15% |

The lightweight Student CE model is used for the
public application demonstration.

## Model Efficiency

| Metric | Teacher | Student CE |
|---|---:|---:|
| Parameters | 23.53M | 458,667 |
| Checkpoint Size | 90.05 MB | 1.83 MB |
| CPU Throughput | 22.28 FPS | 86.35 FPS |

## Technologies Used

- Python
- PyTorch
- Torchvision
- Streamlit
- NumPy
- Pillow
- Matplotlib
- Scikit-learn

## Run Locally

Install dependencies:

pip install -r requirements.txt

Run the application:

streamlit run app.py

## Project Purpose

The project investigates the trade-off between
classification performance and computational efficiency
for waste classification.

## Limitations

The model is an academic prototype and should not be
used as an autonomous industrial waste-sorting system
without further validation.
