# ==============================================
# Check structure and contents of lux_communes_kpi.geojson
# ==============================================

import geopandas as gpd
import pandas as pd

# パスを適宜調整
path = "outputs/lux_communes_kpi.geojson"

# 読み込み
gdf = gpd.read_file(path)
gdf = gdf.to_crs(4326)

print("===============================================")
print("✅ Basic Info")
print("===============================================")
print(gdf.info())

print("\n===============================================")
print("📍 Commune samples (first 10 rows)")
print("===============================================")
print(gdf[["commune"]].head(10))

print("\n===============================================")
print("📊 Columns overview")
print("===============================================")
print(list(gdf.columns))

# 数値カラムのみ統計表示
num_cols = [c for c in gdf.columns if gdf[c].dtype != "object"]
print("\n===============================================")
print("📈 Summary statistics (numerical columns)")
print("===============================================")
print(gdf[num_cols].describe().round(2))

# 欠損値確認
print("\n===============================================")
print("⚠️ Missing values per column")
print("===============================================")
print(gdf.isna().sum())

# 高齢化・外国人比率など主要指標のサンプル確認
print("\n===============================================")
print("🧩 Sample key indicators (first 5 communes)")
print("===============================================")
key_cols = [c for c in ["commune", "share_65p", "share_foreign_citizenship", 
                        "share_not_in_family_nucleus", "old_dep_ratio", "youth_dep_ratio"] if c in gdf.columns]
print(gdf[key_cols].head())