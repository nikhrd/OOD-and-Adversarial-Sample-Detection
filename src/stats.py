import torch
import numpy as np
import config
from src.hooks import captured_features

def get_class_stats(dataloader, model, layers):
    all_features = {i: [] for i in range(len(layers))}
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(config.DEVICE)
            captured_features.clear()

            _ = model(inputs)

            for i, feat in enumerate(captured_features):
                all_features[i].append(feat.cpu())

            all_labels.append(labels)

    labels = torch.cat(all_labels)

    stats_per_layer = []

    for i in range(len(layers)):
        layer_feats = torch.cat(all_features[i])

        means = []
        for c in range(config.NUM_CLASSES):
            mask = (labels == c)
            means.append(layer_feats[mask].mean(0))

        means = torch.stack(means)

        zero_mean = layer_feats.clone()
        for c in range(config.NUM_CLASSES):
            zero_mean[labels == c] -= means[c]

        cov = torch.mm(zero_mean.t(), zero_mean) / len(layer_feats)
        precision = torch.linalg.inv(cov + 1e-6 * torch.eye(cov.shape[0]))

        stats_per_layer.append({
            "means": means,
            "precision": precision
        })

    return stats_per_layer