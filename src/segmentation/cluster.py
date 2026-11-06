"""K-Means segmentation (k from silhouette + minimum segment size, elbow logged), cross-checked with a GMM."""

from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, davies_bouldin_score, silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.utils import get_logger

log = get_logger(__name__)

# Weights on standardised cluster centroids that express each business persona.
PERSONAS = {
    "Champions": {"recency_days": -1, "frequency": 1, "monetary_total": 1},
    "Loyal Regulars": {"frequency": 1, "recency_days": -0.5, "avg_order_value": -0.5},
    "Big-Ticket Wholesalers": {"avg_order_value": 2, "monetary_total": 0.5},
    "Seasonal Gift Shoppers": {"q4_share": 2, "frequency": -0.5},
    "Lapsed High-Value": {"recency_days": 1, "monetary_total": 1},
    "Fading Occasionals": {"recency_days": 1, "frequency": -1, "monetary_total": -1},
    "New & Promising": {"tenure_days": -2, "recency_days": -1},
}


def _quality(Z: np.ndarray, labels: np.ndarray, seed: int) -> dict[str, float]:
    return {
        "silhouette": float(silhouette_score(Z, labels, sample_size=min(len(Z), 4000), random_state=seed)),
        "davies_bouldin": float(davies_bouldin_score(Z, labels)),
    }


def name_clusters(centroids: pd.DataFrame) -> dict[int, str]:
    """Assign each cluster a unique persona by maximising total persona fit (Hungarian assignment)."""
    names = list(PERSONAS)
    fit = np.array(
        [[sum(w * row.get(f, 0) for f, w in PERSONAS[n].items()) for n in names] for _, row in centroids.iterrows()]
    )
    rows, cols = linear_sum_assignment(-fit)
    return {int(centroids.index[r]): names[c] for r, c in zip(rows, cols, strict=True)}


def fit_segments(features: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    """Fit scaler, K-Means, GMM and PCA; return the model bundle plus a diagnostics report."""
    sc, seed = cfg["segmentation"], cfg["seed"]
    X = features[sc["features"]].fillna(0).clip(lower=0)
    prep = make_pipeline(FunctionTransformer(np.log1p, feature_names_out="one-to-one"), StandardScaler())
    Z = prep.fit_transform(X)
    ks = range(sc["k_range"][0], min(sc["k_range"][1], len(PERSONAS)) + 1)
    kmeans_runs = {k: KMeans(k, n_init=10, random_state=seed).fit(Z) for k in ks}
    elbow = [{"k": k, "inertia": float(m.inertia_), **_quality(Z, m.labels_, seed)} for k, m in kmeans_runs.items()]
    for r in elbow:
        r["min_share"] = float(np.bincount(kmeans_runs[r["k"]].labels_).min() / len(Z))
    ok = [r for r in elbow if r["min_share"] >= sc["min_cluster_share"]] or elbow
    best = max(r["silhouette"] for r in ok)
    best_k = max(r["k"] for r in ok if r["silhouette"] >= best - sc["silhouette_tolerance"])
    kmeans = kmeans_runs[best_k]
    gmms = {k: GaussianMixture(k, covariance_type="full", random_state=seed).fit(Z) for k in ks}
    gmm_k = min(gmms, key=lambda k: gmms[k].bic(Z))
    gmm_labels = gmms[gmm_k].predict(Z)
    centroids = pd.DataFrame(kmeans.cluster_centers_, columns=sc["features"])
    report = {
        "k": best_k,
        "elbow": elbow,
        "kmeans": _quality(Z, kmeans.labels_, seed),
        "gmm": {
            "k": gmm_k,
            "bic": [{"k": k, "bic": float(g.bic(Z))} for k, g in gmms.items()],
            **_quality(Z, gmm_labels, seed),
            "ari_vs_kmeans": float(adjusted_rand_score(kmeans.labels_, gmm_labels)),
        },
    }
    log.info("segmentation fitted", extra={"k": best_k, "silhouette": report["kmeans"]["silhouette"], "gmm_k": gmm_k})
    return {
        "prep": prep,
        "kmeans": kmeans,
        "pca": PCA(2, random_state=seed).fit(Z),
        "names": name_clusters(centroids),
        "report": report,
    }


def assign(features: pd.DataFrame, bundle: dict[str, Any], cfg: dict[str, Any]) -> pd.DataFrame:
    """Segment label and 2-D PCA coordinates for each customer."""
    prep: Pipeline = bundle["prep"]
    Z = prep.transform(features[cfg["segmentation"]["features"]].fillna(0).clip(lower=0))
    ids = bundle["kmeans"].predict(Z)
    xy = bundle["pca"].transform(Z)
    return pd.DataFrame(
        {"segment_id": ids, "segment": [bundle["names"][i] for i in ids], "pca_x": xy[:, 0], "pca_y": xy[:, 1]},
        index=features.index,
    )
