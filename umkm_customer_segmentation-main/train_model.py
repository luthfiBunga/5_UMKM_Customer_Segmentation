
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans

from src.features import (
    CATEGORICAL_FEATURES,
    CLUSTER_NAMES,
    FEATURES,
    NUMERIC_FEATURES,
)

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "data_pelanggan_clean.csv"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH)

missing_columns = [col for col in FEATURES if col not in df.columns]

if missing_columns:
    raise ValueError(f"Kolom tidak ditemukan: {missing_columns}")

X = df[FEATURES].copy()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numerik",
            Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]),
            NUMERIC_FEATURES,
        ),
        (
            "kategorikal",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]),
            CATEGORICAL_FEATURES,
        ),
    ]
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("kmeans", KMeans(n_clusters=4, random_state=42, n_init=10)),
])

model.fit(X)

kmeans = model.named_steps["kmeans"]
labels = kmeans.labels_.copy()
profiles = df.assign(cluster=labels).groupby("cluster")[NUMERIC_FEATURES].median()

# Keep cluster IDs aligned with the final notebook's segment definitions.
inactive_cluster = profiles["hari_sejak_transaksi_terakhir"].idxmax()
remaining_clusters = [cluster for cluster in profiles.index if cluster != inactive_cluster]
promo_cluster = profiles.loc[remaining_clusters, "persen_pakai_promo"].idxmax()
remaining_clusters.remove(promo_cluster)
high_value_cluster = profiles.loc[remaining_clusters, "jumlah_transaksi_6bln"].idxmin()
loyal_cluster = next(
    cluster for cluster in remaining_clusters if cluster != high_value_cluster
)
cluster_order = [
    loyal_cluster,
    inactive_cluster,
    promo_cluster,
    high_value_cluster,
]
cluster_mapping = {
    original: canonical
    for canonical, original in enumerate(cluster_order)
}
kmeans.cluster_centers_ = kmeans.cluster_centers_[cluster_order]
kmeans.labels_ = np.array(
    [cluster_mapping[label] for label in labels],
    dtype=int,
)

processed_features = model.named_steps["preprocessor"].transform(X)
score = silhouette_score(
    processed_features,
    kmeans.labels_,
    sample_size=3000,
    random_state=42,
)

joblib.dump(model, ARTIFACTS_DIR / "model.joblib")

metadata = {
    "model_name": "UMKM Customer Segmentation",
    "algorithm": "KMeans",
    "n_clusters": 4,
    "features": FEATURES,
    "numeric_features": NUMERIC_FEATURES,
    "categorical_features": CATEGORICAL_FEATURES,
    "cluster_names": CLUSTER_NAMES,
    "silhouette_score": score,
    "n_samples": len(df),
    "preprocessing": {
        "numeric_imputation": "median",
        "numeric_scaling": "standard",
        "categorical_imputation": "most_frequent",
        "categorical_encoding": "one_hot",
    },
    "random_state": 42,
    "n_init": 10,
    "sklearn_version": sklearn.__version__
}

with (ARTIFACTS_DIR / "metadata.json").open("w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

print("Model berhasil dilatih!")
print("Jumlah data:", len(df))
print("Jumlah fitur numerik:", len(NUMERIC_FEATURES))
print("Jumlah fitur kategorikal:", len(CATEGORICAL_FEATURES))
print("Silhouette score:", round(score, 4))
print("Jumlah pelanggan per cluster:")
print(pd.Series(kmeans.labels_).value_counts().sort_index().to_string())
print("Model tersimpan di", ARTIFACTS_DIR / "model.joblib")
print("Metadata tersimpan di", ARTIFACTS_DIR / "metadata.json")
