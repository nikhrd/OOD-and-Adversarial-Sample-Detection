import torch

DEVICE = torch.device("cpu")  # FORCE CPU

BATCH_SIZE = 64
NUM_CLASSES = 10
IMAGE_SIZE = 224

DATA_PATH = "./data"
ARTIFACT_PATH = "artifacts/detector_artifacts.pkl"