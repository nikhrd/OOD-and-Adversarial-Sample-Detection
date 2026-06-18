import torch
import numpy as np
import config
from src.hooks import captured_features

def get_mahalanobis_score(inputs, model, stats, magnitude=0.002):
    inputs = inputs.to(config.DEVICE).requires_grad_(True)

    captured_features.clear()
    logits = model(inputs)[:, :config.NUM_CLASSES]
    best_class = logits.argmax(1)

    features = captured_features[-1]
    means = stats[-1]["means"].to(config.DEVICE)
    precision = stats[-1]["precision"].to(config.DEVICE)

    target_means = means[best_class]

    diff = features - target_means
    intermediate = torch.mm(diff, precision)
    distance = torch.sum(intermediate * diff, dim=1)

    loss = torch.mean(-0.5 * distance)
    model.zero_grad()
    loss.backward()

    gradient = inputs.grad.data.sign()
    perturbed = inputs.data - magnitude * gradient

    return perturbed


def calculate_layer_scores(inputs, model, stats):
    model.eval()
    captured_features.clear()

    with torch.no_grad():
        _ = model(inputs)

    layer_scores = []

    for i, feat in enumerate(captured_features):
        means = stats[i]["means"].to(config.DEVICE)
        precision = stats[i]["precision"].to(config.DEVICE)

        batch_scores = []

        for j in range(feat.size(0)):
            diff = feat[j] - means
            dist = torch.matmul(torch.matmul(diff, precision), diff.t()).diag()
            batch_scores.append(-torch.min(dist).item())

        layer_scores.append(batch_scores)

    return np.array(layer_scores).T