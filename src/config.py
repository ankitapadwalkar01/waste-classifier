from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
REPORT_DIR = PROJECT_ROOT / "reports"
SPLIT_FILE = DATA_DIR / "splits.csv"

DATASET_URL = "https://archive.ics.uci.edu/static/public/908/realwaste.zip"
DATASET_ID = 908
DATASET_NAME = "RealWaste"

CLASS_NAMES = [
    "Cardboard",
    "Food Organics",
    "Glass",
    "Metal",
    "Miscellaneous Trash",
    "Paper",
    "Plastic",
    "Textile Trash",
    "Vegetation",
]

IMAGE_SIZE = 160
BATCH_SIZE = 24
NUM_WORKERS = 2
SEED = 42
TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Knowledge-distillation hyperparameters.
KD_TEMPERATURE = 4.0
KD_ALPHA = 0.50  # Weight of ground-truth cross-entropy; remaining weight is KD loss.

# Use pretrained ImageNet weights by default. Pass --no-pretrained to train offline.
PRETRAINED = True
