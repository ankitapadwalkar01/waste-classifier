# WasteWise AI: Lightweight CNN with Knowledge Distillation for Real-Time Waste Classification

A fully software-based Python deep-learning project. It trains a ResNet-50 teacher, a custom lightweight MobileNet-style student (inverted residual blocks, depthwise convolutions, and squeeze-and-excitation attention), and a second copy of the same student using knowledge distillation. It then compares all three models on an untouched test split and provides a Streamlit image-classification demo with Grad-CAM.

## Why this project is stronger than a basic image classifier

- Uses **RealWaste**, a public nine-category landfill-image dataset rather than a two-class toy dataset.
- Implements a **custom lightweight CNN** using inverted residual blocks, depthwise-separable convolution, squeeze-and-excitation attention, and residual shortcuts (458,667 parameters in this implementation).
- Runs a controlled **three-model experiment**: ResNet-50 teacher, student trained with cross-entropy only, and the same student trained with cross-entropy plus teacher-guided distillation.
- Gives both predictive metrics and deployment metrics: macro-F1, per-class precision/recall, confusion matrices, parameter count, checkpoint size, median batch-one latency, and estimated FPS.
- Includes image augmentation, class-balanced sampling, stratified splits, fixed random seeds, early stopping, an inference application, and Grad-CAM visual explanations.
- Generates a project report from actual measured results. It does not invent benchmark scores.

## 1. Project requirements

Recommended: Python 3.10–3.12. A CUDA-capable GPU is optional. The scripts automatically use CUDA if available and otherwise run on CPU. Training on CPU is slower; use the reduced command below for a quick demonstration.

The first teacher run normally downloads pretrained ImageNet weights from torchvision. Use `--no-pretrained` when internet access is unavailable (teacher quality may be lower).

## 2. Setup on Windows (PowerShell)

Open PowerShell in the extracted project folder.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run this once in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

For GPU use, install the PyTorch build matching the CUDA version on the machine by following the official selector at https://pytorch.org/get-started/locally/ before installing the remaining packages. Do not install a CPU-only build and expect CUDA acceleration.

## 3. Download the dataset

```powershell
python download_dataset.py
```

The script downloads the official UCI RealWaste ZIP (approximately 657 MB), safely extracts it under `data/raw/extracted`, and finds the nine-class image directory automatically. An internet connection and roughly 1.5 GB of free disk space are recommended for the archive, extracted files, and model artifacts.

Dataset source: https://archive.ics.uci.edu/dataset/908/realwaste  
Dataset DOI: https://doi.org/10.24432/C5SS4G  
License shown by UCI: CC BY 4.0. Give attribution to the dataset creators when redistributing results or the dataset.

## 4. Train the complete experiment

```powershell
python train.py
```

Default settings: 160 × 160 input, batch size 24, up to 6 teacher epochs and 12 epochs for each student. Early stopping may stop a model earlier. The test partition is not used for model selection; it is only evaluated after training.

For a lower-compute run on a CPU or laptop:

```powershell
python train.py --teacher-epochs 2 --student-epochs 3 --batch-size 8 --workers 0 --image-size 128
```

This is a reduced-epoch run, not the strongest final experiment. It still uses pretrained teacher weights. For a fully offline run without the ImageNet weight download, add `--no-pretrained`; the teacher is then trained from scratch and may be substantially weaker. For the report submission, use the normal training command when resources permit.

To force a fresh stratified 70/15/15 split:

```powershell
python train.py --rebuild-splits
```

The split is saved in `data/splits.csv`, so repeated runs use the same examples unless `--rebuild-splits` is specified. Do not change the test split after seeing the test results; that would weaken the validity of the evaluation.

## 5. Launch the application

After training:

```powershell
streamlit run app.py
```

Open the local URL printed by Streamlit. Upload a JPG, JPEG, PNG, or WEBP image. The interface shows the top-three classes, predicted confidence, and a Grad-CAM overlay. The app runs locally; uploaded images are not sent to a remote classification service by this code.

## 6. Generate report and plots

```powershell
python plot_results.py
python generate_report.py
```

The measured project report is written to `reports/PROJECT_REPORT.md`. It reads the saved test metrics, so run training first. `artifacts/performance_comparison.png` contains the accuracy and macro-F1 comparison.

## 7. Project structure

```text
waste_kd_project/
├── app.py                         # Streamlit inference interface
├── download_dataset.py            # UCI RealWaste downloader/extractor
├── train.py                       # Complete teacher/CE/KD training pipeline
├── generate_report.py             # Builds the report from measured results
├── plot_results.py                # Generates comparison plot
├── requirements.txt
├── src/
│   ├── config.py                  # Paths, classes, training constants
│   ├── data.py                    # Image discovery, splits, transforms, loaders
│   ├── models.py                  # Teacher, student, distillation loss
│   ├── engine.py                  # Training and validation loops
│   ├── evaluation.py              # Test metrics, latency, confusion matrices
│   └── inference.py               # Prediction and Grad-CAM
├── data/
│   ├── raw/                       # Download/extract location (not included in ZIP)
│   └── splits.csv                 # Created during first training run
├── artifacts/                     # Checkpoints, logs, evaluation outputs
├── reports/                       # Generated final report
└── tests/                          # Model and loss smoke tests
```

## 8. What each experiment proves

| Experiment | Model | Training objective | Purpose |
|---|---|---|---|
| A | ResNet-50 | Cross-entropy | High-capacity teacher reference |
| B | LightweightWasteCNN | Cross-entropy | Small-model baseline |
| C | Same LightweightWasteCNN | Cross-entropy + KL distillation | Measures the effect of teacher guidance |

The student models share the same architecture and initial weights. The comparison is intended to isolate the addition of the distillation loss. The teacher is not used at inference time; only the small student is loaded by the demo.

## 9. Mathematical basis

Let `z_t` be teacher logits, `z_s` student logits, `y` ground-truth labels, and `T` the softmax temperature:

```text
p_t = softmax(z_t / T)
p_s = softmax(z_s / T)
L_KD = T^2 * KLDiv(log(p_s), p_t)
L_total = alpha * CrossEntropy(z_s, y) + (1 - alpha) * L_KD
```

The temperature exposes relative similarities between classes. For example, the teacher may assign non-zero probability to both paper and cardboard when their visual features overlap. The student learns from these relative probabilities in addition to the hard correct label. The factor `T²` compensates for the gradient scale change caused by temperature scaling.

Depthwise-separable convolution decomposes a dense convolution into a depthwise spatial convolution and a pointwise channel-mixing convolution. The student adds inverted residual expansion/projection, squeeze-and-excitation channel attention, and residual shortcuts. These modules provide a stronger feature extractor while keeping the parameter count far below the ResNet-50 teacher.

## 10. How to interpret the results correctly

Do not claim that knowledge distillation improved performance unless `student_kd` actually performs better than `student_ce` on the metric you cite. It is possible for KD to improve macro-F1 but not accuracy, or to have no benefit on a small dataset. Report the result as measured.

Macro-F1 is important because the dataset has unequal class counts. Accuracy alone can hide poor performance on minority categories. Review the per-class CSV files and confusion matrices rather than relying on one overall score.

The reported FPS is calculated as the inverse of median single-image inference latency after warm-up. Metrics are collected on the selected training device and on CPU, allowing a reproducible CPU comparison even when training uses a GPU. It is a model-only benchmark, not an end-to-end sorting-system speed guarantee.

## 11. Troubleshooting

- **`Could not find a folder containing all nine class folders`:** confirm that the archive extracted fully and that all nine class folders exist under `data/raw/extracted`.
- **Download interruption:** rerun `python download_dataset.py`; delete `data/raw/realwaste.zip.part` if a partial download remains.
- **CUDA out-of-memory:** reduce `--batch-size` to 8 or 4 and consider `--image-size 128`.
- **Slow CPU training:** use the quick command for a smoke test, then run the full experiment on a GPU machine if available.
- **Torchvision pretrained-weight download fails:** retry with internet access, or use `--no-pretrained` and clearly identify that run as training without ImageNet initialization.
- **Streamlit says model missing:** finish `python train.py` before running the app.

## 12. Tests

```powershell
python -m pytest tests -q
```

Tests check student output dimensions, an upper bound on student parameter count, and numerical/gradient validity of the distillation objective.

## 13. References

1. Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning*. Information, 14(12), 633. https://doi.org/10.3390/info14120633
2. Single, S., Iranmanesh, S., & Raad, R. (2023). *RealWaste* [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5SS4G
3. Hinton, G., Vinyals, O., & Dean, J. (2015). *Distilling the Knowledge in a Neural Network*. https://arxiv.org/abs/1503.02531
4. PyTorch Contributors. *Knowledge Distillation Tutorial*. https://docs.pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html
5. Howard, A. G., et al. (2017). *MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications*. https://arxiv.org/abs/1704.04861
