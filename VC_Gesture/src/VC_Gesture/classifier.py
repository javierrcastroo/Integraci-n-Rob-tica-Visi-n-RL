# classifier.py
import numpy as np

def knn_predict(feature_vec, gallery, k=5):
    """
    gallery: lista de (feature_vec_guardado, label)
    """
    if len(gallery) == 0:
        return None, None

    dists = []
    for f_ref, lab in gallery:
        if f_ref.shape != feature_vec.shape:
            continue
        dist = np.linalg.norm(feature_vec - f_ref)
        dists.append((dist, lab))

    if len(dists) == 0:
        return None, None

    dists.sort(key=lambda x: x[0])
    k = min(k, len(dists))
    k_neigh = dists[:k]

    # voto mayoritario
    labels = [lab for _, lab in k_neigh]
    # distancia media
    avg_dist = sum(d for d, _ in k_neigh) / k

    # escoger el label más frecuente
    best_label = max(set(labels), key=labels.count)
    return best_label, avg_dist