# config.py
import torch

DEVICE = torch.device("cpu")  # Force CPU execution

BATCH_SIZE = 64
NUM_CLASSES = 10
IMAGE_SIZE = 224

DATA_PATH = "./data"
ARTIFACT_PATH = "artifacts/detector_artifacts.pkl"
SECRET_KEY_PATH = "artifacts/secret.key"
LOG_FILE_PATH = "artifacts/system.log"