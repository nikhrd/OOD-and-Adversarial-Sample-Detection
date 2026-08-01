# src/faiss_index.py
import faiss
import numpy as np
import torch
import config
from src.hooks import captured_features
from tqdm import tqdm

def build_faiss_index(model, dataloader, num_samples=1000):
    """
    Extracts deep features from clean in-distribution training data
    and builds an L2 FAISS index alongside a distance threshold.
    """
    model.eval()
    embeddings = []
    
    with torch.no_grad():
        for inputs, _ in tqdm(
            dataloader,
            desc="Extracting FAISS Embeddings",
            unit="batch",
            colour="cyan"
        ):
            inputs = inputs.to(config.DEVICE)
            captured_features.clear()
            _ = model(inputs)
            
            # Use top layer feature representations
            top_features = captured_features[-1].cpu().numpy()
            embeddings.append(top_features)
            
            if len(embeddings) * config.BATCH_SIZE >= num_samples:
                break

    embeddings = np.vstack(embeddings).astype('float32')
    
    # Initialize FAISS Flat L2 Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    
    # Compute baseline distance threshold (Mean + 2 * StdDev)
    distances, _ = index.search(embeddings, k=5)
    mean_dist = float(np.mean(distances))
    std_dist = float(np.std(distances))
    threshold = mean_dist + 2.0 * std_dist
    
    return index, threshold

def disambiguate_sample(feature_vector, index, threshold):
    """
    Differentiates between Adversarial and OOD samples using FAISS L2 distance.
    """
    feature_vector = feature_vector.detach().cpu().numpy().astype('float32')
    distances, _ = index.search(feature_vector, k=5)
    avg_distance = float(np.mean(distances))
    
    if avg_distance <= threshold:
        return "Adversarial Attack", avg_distance
    else:
        return "Out-of-Distribution (OOD)", avg_distance