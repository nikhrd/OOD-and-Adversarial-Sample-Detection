import pickle
from src.model import load_model
from src.hooks import register_hooks
from src.utils import get_dataloader
from src.stats import get_class_stats
from src.mahalanobis import get_mahalanobis_score, calculate_layer_scores
from src.adversarial import generate_adversarial_image
from src.ensemble import train_judge
import config
import numpy as np

model = load_model()
layers = register_hooks(model)

loader = get_dataloader()

print("Extracting stats...")
stats = get_class_stats(loader, model, layers)

print("Collecting scores...")
X_clean, X_adv = [], []

for inputs, labels in loader:
    inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)

    clean = get_mahalanobis_score(inputs, model, stats)
    clean_scores = calculate_layer_scores(clean, model, stats)

    adv = generate_adversarial_image(model, inputs, labels)
    adv = get_mahalanobis_score(adv, model, stats)
    adv_scores = calculate_layer_scores(adv, model, stats)

    X_clean.append(clean_scores)
    X_adv.append(adv_scores)

    if len(X_clean) * config.BATCH_SIZE > 500:
        break

X_clean = np.vstack(X_clean)
X_adv = np.vstack(X_adv)

print("Training judge...")
judge = train_judge(X_clean, X_adv)

with open(config.ARTIFACT_PATH, "wb") as f:
    pickle.dump({"stats": stats, "judge": judge}, f)

print(" Pipeline complete!")