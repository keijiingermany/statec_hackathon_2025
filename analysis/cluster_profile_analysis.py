# ==============================================
# Cluster Profile Analysis – Luxembourg Ageing Dashboard
# Author: Keiji Uehara + ChatGPT
# ==============================================

import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

# ---- Load data ----
path = "outputs/lux_communes_kpi.geojson"
gdf = gpd.read_file(path)
gdf = gdf.to_crs(4326)

# ---- Select features (same as dashboard default) ----
features = ["share_65p", "share_foreign_citizenship", "share_not_in_family_nucleus"]

# Drop NA
df = gdf.dropna(subset=features).copy()

# ---- Standardize ----
scaler = StandardScaler()
X = scaler.fit_transform(df[features])

# ---- Cluster (you can change n_clusters=3 if needed) ----
kmeans = KMeans(n_clusters=3, random_state=42, n_init="auto")
df["cluster"] = kmeans.fit_predict(X)

# ---- Compute cluster means ----
cluster_summary = (
    df.groupby("cluster")[features]
    .agg(["mean", "std"])
    .round(2)
)

# ---- Identify representative communes ----
# The one closest to each cluster centroid
centroids = kmeans.cluster_centers_
df_std = pd.DataFrame(X, index=df.index, columns=features)
df_std["cluster"] = df["cluster"]

representatives = []
for i, c in enumerate(centroids):
    cluster_df = df_std[df_std["cluster"] == i]
    # compute Euclidean distance to centroid
    dists = np.linalg.norm(cluster_df[features] - c, axis=1)
    idx = cluster_df.index[np.argmin(dists)]
    representatives.append((i, df.loc[idx, "commune"]))

# ---- Print results ----
print("======================================")
print(" Cluster Profiles (Luxembourg Communes)")
print("======================================\n")

for i, rep_name in representatives:
    mean_vals = cluster_summary.loc[i].xs("mean", level=1)
    print(f"Cluster {i}  |  Representative commune: {rep_name}")
    print(f"  - share_65p: {mean_vals['share_65p']:.2f}%")
    print(f"  - share_foreign_citizenship: {mean_vals['share_foreign_citizenship']:.2f}%")
    print(f"  - share_not_in_family_nucleus: {mean_vals['share_not_in_family_nucleus']:.2f}%")
    print()

print("Cluster summary table:")
print(cluster_summary)