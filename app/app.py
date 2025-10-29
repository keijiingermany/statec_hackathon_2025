# ==============================================
# Luxembourg Ageing Dashboard 2021+ (Enhanced)
# Author: Keiji Uehara (STATEC Hackathon) + ChatGPT
# ==============================================

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import json

# Optional deps for analysis
HAS_STATSMODELS = True
try:
    import statsmodels.api as sm
except Exception:
    HAS_STATSMODELS = False

HAS_SKLEARN = True
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
except Exception:
    HAS_SKLEARN = False

# --------------------------
# 1) Basic settings
# --------------------------
st.set_page_config(
    page_title="Luxembourg Ageing Dashboard – Enhanced",
    layout="wide"
)

# TailwindCSS風 Modern Slate Theme
st.markdown("""
    <style>
        /* ========= BASE ========== */
        body, .stApp {
            background-color: #1e293b; /* slate-800 */
            color: #e2e8f0;            /* slate-200 */
            font-family: 'Inter', 'Segoe UI', Roboto, sans-serif;
        }

        /* ========= SIDEBAR ========== */
        section[data-testid="stSidebar"] {
            background-color: #0f172a; /* slate-900 */
            border-right: 1px solid #334155; /* slate-700 */
        }

        /* ========= HEADINGS ========== */
        h1, h2, h3, h4, h5 {
            color: #38bdf8; /* sky-400 */
            font-weight: 600;
            letter-spacing: 0.4px;
        }

        /* ========= BUTTONS ========== */
        a, .stButton>button {
            background: linear-gradient(90deg, #0ea5e9, #0369a1);
            color: white !important;
            border: none;
            border-radius: 6px;
            padding: 0.4rem 0.8rem;
            transition: all 0.2s ease-in-out;
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #38bdf8, #0284c7);
            transform: translateY(-1px);
        }

        /* ========= CHARTS / PLOTS ========== */
        .plot-container > div {
            background-color: #1e293b !important; /* slate-800 */
            border-radius: 8px;
        }

        /* ========= TABLES ========== */
        table {
            background-color: #0f172a !important;
            color: #e2e8f0 !important;
            border-collapse: collapse;
            border-radius: 6px;
        }
        thead th {
            background-color: #1e293b !important;
            color: #38bdf8 !important;
            font-weight: 600;
            padding: 8px;
        }
        tbody tr:nth-child(even) {
            background-color: #1e293b !important;
        }

        /* ========= INSIGHT CARDS ========== */
        div[data-testid="stMarkdownContainer"] div[style*="box-shadow"] {
            background-color: #0f172a !important; /* slate-900 */
            color: #e2e8f0 !important;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(15, 23, 42, 0.5);
        }
        .stMarkdown h5 {
            color: #38bdf8 !important;
        }

        /* ========= ALERTS / CAPTIONS ========== */
        .stAlert {
            background-color: #1e293b !important;
            border: 1px solid #334155 !important;
            color: #e2e8f0 !important;
        }

        footer, .stCaption {
            color: #94a3b8 !important; /* slate-400 */
        }

        /* ========= SCROLLBAR (optional) ========== */
        ::-webkit-scrollbar {
            width: 8px;
        }
        ::-webkit-scrollbar-thumb {
            background: #475569;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #64748b;
        }

        /* ========= CARD TEXT CONTRAST IMPROVE ========== */
        div[data-testid="stMarkdownContainer"] div[style*="box-shadow"] p {
            color: #e5e7eb !important; /* slightly brighter text for readability */
        }

        /* ========= MAP BORDER ENHANCE ========== */
        .plot-container > div {
            border: 1px solid #334155 !important;
        }

    </style>
""", unsafe_allow_html=True)

# Resolve base path relative to project structure
try:
    base = Path(__file__).resolve().parents[1]
except NameError:
    base = Path.cwd().parents[0]

data_dir = base / "outputs"
geo_path = data_dir / "lux_communes_kpi.geojson"

# --------------------------
# 2) Data loading
# --------------------------
@st.cache_data
def load_data():
    gdf = gpd.read_file(geo_path)
    # CRS to WGS84
    gdf = gdf.to_crs(4326)

    # Prepare centroids for map centering
    gdf["centroid_lon"] = gdf.geometry.centroid.x
    gdf["centroid_lat"] = gdf.geometry.centroid.y

    # Normalize commune column if needed
    if "commune" not in gdf.columns:
        for c in ["COMMUNE", "commune_y", "commune_x"]:
            if c in gdf.columns:
                gdf = gdf.rename(columns={c: "commune"})
                break
        # Drop potential duplicate
        if "commune_x" in gdf.columns:
            gdf = gdf.drop(columns=["commune_x"], errors="ignore")
        if "commune_y" in gdf.columns:
            gdf = gdf.drop(columns=["commune_y"], errors="ignore")

    # If 'year' not present, create a constant 2021 to keep UI consistent
    if "year" not in gdf.columns:
        gdf["year"] = 2021

    # Ensure numeric columns are numeric
    for col in gdf.columns:
        if col.startswith(("share_", "pop_", "old_dep_ratio", "youth_dep_ratio")):
            gdf[col] = pd.to_numeric(gdf[col], errors="coerce")

    return gdf

gdf = load_data()

@st.cache_data
def get_geojson(_gdf):
    """Return GeoJSON string for a GeoDataFrame.
    Using an underscore to avoid Streamlit hashing on unhashable objects.
    """
    return _gdf.to_json()

if gdf.empty:
    st.error("❌ Could not load GeoJSON data. Please run preprocessing to generate the GeoJSON first.")
    st.stop()

geojson_str = get_geojson(gdf)

# --------------------------
# 3) KPI dictionary
# --------------------------
kpi_options = {
    "share_65p": "Share of Population Aged 65+ (%)",
    "share_foreign_citizenship": "Share of Foreign Citizens (%)",
    "share_not_in_family_nucleus": "Share Not in Family Nucleus (%)",
    "old_dep_ratio": "Old-Age Dependency Ratio (%)",
    "youth_dep_ratio": "Youth Dependency Ratio (%)",
    "elderly_isolation_index": "Elderly Isolation Index (65+ × Non-Family %)",
}

kpi_descriptions = {
    "share_65p": "Share of the population aged 65 and over. A direct indicator of ageing.",
    "share_foreign_citizenship": "Share of foreign citizens. Indicates immigrant composition which can affect local demographics.",
    "share_not_in_family_nucleus": "Share not living in a family nucleus. Proxy for single-person or non-traditional households.",
    "old_dep_ratio": "Old-age dependency ratio (older population / working-age population). Rough indicator of social support burden.",
    "youth_dep_ratio": "Youth dependency ratio (young population / working-age population).",
    "elderly_isolation_index": (
        "Composite index combining ageing and social isolation. "
        "Higher values indicate communes with more elderly living alone risk."
    ),
}

# --------------------------
# 4) Sidebar controls (minimal, no commune picker)
# --------------------------
st.sidebar.header("🗺️ KPI Selection & Analysis")

# Year slider (if multiple years exist)
years = sorted(gdf["year"].dropna().unique().tolist())
if len(years) > 1:
    selected_year = st.sidebar.slider("Select year", min_value=int(min(years)), max_value=int(max(years)), value=int(max(years)))
else:
    selected_year = years[0]
subset = gdf[gdf["year"] == selected_year].copy()

# --------------------------
# Add derived KPI: Elderly Isolation Index
# --------------------------
if "share_65p" in subset.columns and "share_not_in_family_nucleus" in subset.columns:
    subset["elderly_isolation_index"] = (
        subset["share_65p"] * subset["share_not_in_family_nucleus"] / 100
    )
else:
    subset["elderly_isolation_index"] = None

# KPI selectors
selected_kpi = st.sidebar.selectbox(
    "Primary KPI (map color):",
    options=list(kpi_options.keys()),
    format_func=lambda x: kpi_options[x]
)

compare_options = ["(None)"] + list(kpi_options.keys())
selected_kpi_2 = st.sidebar.selectbox(
    "Secondary KPI (for scatter/regression):",
    options=compare_options,
    index=0,
    format_func=lambda x: "—" if x == "(None)" else kpi_options[x]
)

# Cluster options
with st.sidebar.expander("🧠 Clustering (optional)"):
    enable_cluster = st.checkbox("Enable commune clustering", value=False)
    cluster_features = st.multiselect(
        "Features for clustering",
        options=list(kpi_options.keys()),
        default=["share_65p", "share_foreign_citizenship", "share_not_in_family_nucleus"]
    )
    n_clusters = st.slider("Number of clusters", min_value=2, max_value=6, value=3, step=1)

# --------------------------
# 5) Title & layout start
# --------------------------

# ページ全体のタイトル（少し控えめ）
st.markdown("<h1 style='margin-bottom:0.3em;'>🏙️ Luxembourg Ageing Dashboard 2021+</h1>", unsafe_allow_html=True)
st.markdown("<hr style='margin-top:0.2em; margin-bottom:1em; border:1px solid #334155;' />", unsafe_allow_html=True)


# --------------------------
# 6) Build the choropleth map
# --------------------------
def make_map(df, color_col, color_label):
    fig = px.choropleth_mapbox(
        df,
        geojson=json.loads(geojson_str),
        locations=df.index,
        color=color_col,
        hover_name="commune",
        hover_data={
            "pop_total_2021": True if "pop_total_2021" in df.columns else False,
            "share_65p": True if "share_65p" in df.columns else False,
            "share_foreign_citizenship": True if "share_foreign_citizenship" in df.columns else False,
            "share_not_in_family_nucleus": True if "share_not_in_family_nucleus" in df.columns else False,
            "old_dep_ratio": True if "old_dep_ratio" in df.columns else False,
            "youth_dep_ratio": True if "youth_dep_ratio" in df.columns else False,
        },
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        opacity=0.85,
        height=700
    )

    # ← この部分を追加（ルクセンブルク中心にズーム）
    fig.update_layout(
        mapbox_center={"lat": 49.815, "lon": 6.13},
        mapbox_zoom=8.5,
        margin={"r":0, "t":20, "l":0, "b":0},  # ← topに20px統一
        coloraxis_colorbar=dict(title=color_label)
    )

    return fig

map_fig = make_map(subset, selected_kpi, kpi_options[selected_kpi])

# --------------------------
# 7) Scatter + Regression (if selected_kpi_2 chosen)
# --------------------------
def make_scatter_and_stats(df, xcol, ycol):
    stats_info = {}
    df2 = df.dropna(subset=[xcol, ycol]).copy()

    if HAS_STATSMODELS and not df2.empty:
        X = sm.add_constant(df2[xcol])
        y = df2[ycol]
        model = sm.OLS(y, X).fit()
        stats_info["r2"] = float(model.rsquared)
        try:
            stats_info["slope"] = float(model.params.get(xcol, float("nan")))
            stats_info["intercept"] = float(model.params.get("const", float("nan")))
        except Exception:
            stats_info["slope"] = float("nan")
            stats_info["intercept"] = float("nan")
        stats_info["n"] = int(len(df2))

    scatter_kwargs = dict(
        data_frame=df2,
        x=xcol,
        y=ycol,
        hover_name="commune",
        labels={
            xcol: kpi_options.get(xcol, xcol),
            ycol: kpi_options.get(ycol, ycol),
        }
    )
    try:
        scatter_fig = px.scatter(**scatter_kwargs, trendline="ols")
    except Exception:
        scatter_fig = px.scatter(**scatter_kwargs)

    scatter_fig.update_layout(
        margin={"r":0,"t":20,"l":0,"b":0},
        height=700,
        xaxis_title=kpi_options.get(xcol, xcol),
        yaxis_title=kpi_options.get(ycol, ycol),
    )
    return scatter_fig, stats_info

# --------------------------
# 7.5) Interactive Correlation Heatmap (Plotly-based)
# --------------------------
def make_correlation_heatmap(df, selected_features):
    import numpy as np
    import plotly.express as px

    if not selected_features or len(selected_features) < 2:
        st.warning("⚠️ Please select at least two KPIs from the sidebar to generate a correlation matrix.")
        return

    # 相関計算
    df_kpi = df[selected_features].fillna(0)
    corr = df_kpi.corr(method="pearson")

    # 絶対値による強調（強相関ハイライト）
    abs_corr = corr.abs()
    mask = abs_corr >= 0.6

    # Plotlyヒートマップ
    fig = px.imshow(
        corr,
        x=[kpi_options.get(c, c) for c in corr.columns],
        y=[kpi_options.get(c, c) for c in corr.index],
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        aspect="auto"
    )

    # セルのハイライト（強相関部分を太枠で囲む）
    for i, row in enumerate(corr.index):
        for j, col in enumerate(corr.columns):
            if mask.iloc[i, j] and i != j:
                fig.add_shape(
                    type="rect",
                    x0=j-0.5, x1=j+0.5,
                    y0=i-0.5, y1=i+0.5,
                    line=dict(color="black", width=2)
                )

    fig.update_layout(
        title="Correlation Heatmap (Selected KPIs)",
        xaxis_title="",
        yaxis_title="",
        coloraxis_colorbar=dict(title="Pearson R"),
        height=550,
        margin=dict(l=30, r=30, t=60, b=30)
    )

    # インタラクティブクリックイベント（Scatterへジャンプ）
    clicked = st.plotly_chart(fig, use_container_width=True, on_select="rerun")

    # Scatter自動切替：選択ペアをSidebar変数に格納
    if clicked and "points" in clicked.selection and len(clicked.selection["points"]) > 0:
        pt = clicked.selection["points"][0]
        x_label, y_label = corr.columns[pt["x"]], corr.index[pt["y"]]
        st.session_state["selected_kpi"] = x_label
        st.session_state["selected_kpi_2"] = y_label
        st.rerun()

# --------------------------
# 8) Clustering (optional)
# --------------------------
def run_clustering(df, features, n_clusters=3):
    if not HAS_SKLEARN:
        st.warning("scikit-learn not found; clustering is disabled. Run pip install scikit-learn to enable.")
        return df, None

    feats = [f for f in features if f in df.columns]
    work = df[feats].dropna().copy()
    if work.empty:
        st.warning("No data available for the selected features to cluster.")
        return df, None

    scaler = StandardScaler()
    X = scaler.fit_transform(work[feats])
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X)

    df_out = df.copy()
    df_out["cluster"] = None
    df_out.loc[work.index, "cluster"] = labels

    profiles = work.assign(cluster=labels).groupby("cluster").mean().reset_index()
    profiles = profiles.sort_values("cluster").reset_index(drop=True)  # ← クラスタ順序固定
    return df_out, profiles

# --------------------------
# 9) Insight generator（カードUI対応）
# --------------------------
def generate_insight(df, kpi1, kpi2, stats=None, cluster_profiles=None):
    """Return insights as a dictionary instead of plain text."""
    insights = {}

    # --- Top & Bottom communes ---
    tmp = df.dropna(subset=[kpi1]).copy()
    if not tmp.empty:
        top_row = tmp.nlargest(1, kpi1).iloc[0]
        bot_row = tmp.nsmallest(1, kpi1).iloc[0]
        insights["Highest"] = f"{top_row['commune']} ({top_row[kpi1]:.2f})"
        insights["Lowest"] = f"{bot_row['commune']} ({bot_row[kpi1]:.2f})"

    # --- Correlation info ---
    if kpi2 and kpi2 != "(None)" and stats and "r2" in stats:
        slope = stats.get("slope", float("nan"))
        r2 = stats.get("r2", float("nan"))
        trend = "⬆️ Increases" if slope > 0 else "⬇️ Decreases"
        insights["📈 Relationship"] = f"{trend} (R²={r2:.2f})"

    return insights

# --------------------------
# 🎨 Visual Insight Cards
# --------------------------
def show_insight_cards(insight_dict):
    """Display insights as neat cards in multiple columns."""
    st.markdown("### 💡 Highlights")

    if not insight_dict:
        st.info("No insights available for the current selection.")
        return

    cols = st.columns(min(3, len(insight_dict)))  # 横に最大3カード

    for i, (title, detail) in enumerate(insight_dict.items()):
        with cols[i % 3]:
            st.markdown(
                f"""
                <div style='background-color:#f8f9fa;
                            border-radius:12px;
                            box-shadow:0 2px 6px rgba(0,0,0,0.1);
                            padding:14px 18px;
                            margin:8px 0;
                            min-height:100px'>
                    <h5 style='margin:0; color:#007bff;'>{title}</h5>
                    <hr style='margin:6px 0; border:0; border-top:1px solid #ddd;'/>
                    <p style='font-size:14px; color:#333; margin:0;'>{detail}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

# --------------------------
# 10) Layout – LEFT COLUMN (Map + Cluster Overview)
# --------------------------
left, right = st.columns([1.5, 1], vertical_alignment="top")

with left:
    # ✅ 左側タイトル（動的KPI）
    st.markdown(
        f"<h3 style='margin-top:0; margin-bottom:0.5em;'>🗺️ {kpi_options[selected_kpi]} ({int(selected_year)})</h3>",
        unsafe_allow_html=True
    )

    # ======================================================
    # 🗺️ MAP SECTION
    # ======================================================
    if enable_cluster:
        # --- クラスタリングON ---
        clustered, cluster_profiles = run_clustering(subset, cluster_features, n_clusters=n_clusters)
        if clustered is not None and "cluster" in clustered.columns:
            map_fig = px.choropleth_mapbox(
                clustered,
                geojson=json.loads(geojson_str),
                locations=clustered.index,
                color="cluster",
                hover_name="commune",
                hover_data={col: True for col in cluster_features if col in clustered.columns},
                mapbox_style="carto-positron",
                opacity=0.85,
                height=700,
                color_continuous_scale="Set2"
            )
            map_fig.update_layout(
                mapbox_center={"lat": 49.815, "lon": 6.13},
                mapbox_zoom=8,
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                coloraxis_colorbar=dict(title="Cluster #")
            )
            st.plotly_chart(map_fig, use_container_width=True, key="main_map")
        else:
            st.warning("⚠️ Clustering failed or no valid data to visualize.")
    else:
        # --- クラスタリングOFF（通常マップ） ---
        map_fig = make_map(subset, selected_kpi, kpi_options[selected_kpi])
        st.plotly_chart(map_fig, use_container_width=True, key="main_map")

    # ======================================================
    # 🧩 CLUSTER OVERVIEW TABLE
    # ======================================================
    if enable_cluster and cluster_profiles is not None and not cluster_profiles.empty:
        st.markdown("### 🧩 Cluster Overview by Commune Type")
        st.caption("Each row summarizes one cluster. Colors correspond exactly to map legend.")

        # --- 自動クラスタラベル生成 ---
        def generate_cluster_label(row):
            """クラスタの特徴に応じた自動ラベル生成（重複回避＋強度インジケータ付き）"""
            mean_vals = subset[cluster_features].mean()

            score_65 = row.get("share_65p", 0) - mean_vals.get("share_65p", 0)
            score_foreign = row.get("share_foreign_citizenship", 0) - mean_vals.get("share_foreign_citizenship", 0)
            score_nonfam = row.get("share_not_in_family_nucleus", 0) - mean_vals.get("share_not_in_family_nucleus", 0)

            # メイン特徴カテゴリ
            if score_65 > 0.05 and score_nonfam > 0.05:
                base_label = "🧓 Aged & Isolated"
            elif score_foreign > 0.05 and score_65 < 0:
                base_label = "🌍 Diverse & Younger"
            elif score_65 < 0 and score_foreign < 0:
                base_label = "🏡 Local & Family-Oriented"
            else:
                base_label = "⚖️ Mixed Profile"

            # 特徴強度ラベル
            magnitude = abs(score_65) + abs(score_foreign) + abs(score_nonfam)
            if magnitude > 0.15:
                strength = " (Strong)"
            elif magnitude > 0.05:
                strength = " (Moderate)"
            else:
                strength = " (Mild)"

            return f"{base_label}{strength}"

        cluster_profiles["Cluster Label"] = cluster_profiles.apply(generate_cluster_label, axis=1)

        # --- テキスト説明生成 ---
        cluster_profiles["Description"] = cluster_profiles.apply(
            lambda r: (
                f"Older pop: {r['share_65p']:.1f}%, "
                f"Foreign: {r['share_foreign_citizenship']:.1f}%, "
                f"Non-family: {r['share_not_in_family_nucleus']:.1f}%"
            ),
            axis=1
        )

        # --- クラスタ番号順に並べ替え（地図と一致させる） ---
        cluster_profiles = cluster_profiles.sort_values("cluster").reset_index(drop=True)

        # --- 表形式データ作成（存在する列のみ選択） ---
        base_cols = [
            "cluster",
            "Cluster Label",
            "share_65p",
            "share_foreign_citizenship",
            "share_not_in_family_nucleus",
            "old_dep_ratio",
            "youth_dep_ratio",
        ]
        available_cols = [c for c in base_cols if c in cluster_profiles.columns]
        display_df = cluster_profiles[available_cols].copy()
        display_df = display_df.rename(columns={
            "cluster": "Cluster #",
            "share_65p": "65+ (%)",
            "share_foreign_citizenship": "Foreign (%)",
            "share_not_in_family_nucleus": "Non-family (%)",
            "old_dep_ratio": "Old Dep. Ratio",
            "youth_dep_ratio": "Youth Dep. Ratio"
        })

        # --- 色づけ（Mapのカラースケールと統一） ---
        color_map = px.colors.qualitative.Set2
        def cluster_color_html(row):
            color = color_map[int(row["Cluster #"]) % len(color_map)]
            label = row["Cluster Label"]
            return f"<span style='background-color:{color};padding:4px 8px;border-radius:6px;color:black;font-weight:600;'>{label}</span>"

        display_df["Profile"] = display_df.apply(cluster_color_html, axis=1)

        # --- テーブル整形表示 ---
        st.markdown(display_df[[
            col for col in ["Cluster #", "Profile", "65+ (%)", "Foreign (%)", "Non-family (%)", "Old Dep. Ratio", "Youth Dep. Ratio"]
            if col in display_df.columns
        ]].to_html(escape=False, index=False), unsafe_allow_html=True)

        # 🔹 英語の凡例説明
        st.markdown("""
        <div style='font-size:13px;color:#94a3b8;margin-top:10px;'>
        🧓 <b>Aged & Isolated</b>: High ageing rate and many single-person households<br>
        🌍 <b>Diverse & Younger</b>: Higher share of foreigners and younger population<br>
        🏡 <b>Local & Family-Oriented</b>: Lower ageing and foreign population – stable family areas<br>
        ⚖️ <b>Mixed Profile</b>: No dominant demographic pattern
        </div>
        """, unsafe_allow_html=True)

with right:
    # --- Comparative Analysis ---
    st.markdown("### 🔍 Comparative Analysis")
    stats_for_msg = None
    if selected_kpi_2 != "(None)" and selected_kpi_2 != selected_kpi:
        scatter_fig, stats_for_msg = make_scatter_and_stats(subset, selected_kpi, selected_kpi_2)
        st.plotly_chart(scatter_fig, use_container_width=True, key="scatter_chart")
        if not HAS_STATSMODELS:
            st.caption("Install statsmodels to enable regression statistics (slope, R²).")

    # --- Highlights ---
    cluster_profiles = None
    if enable_cluster:
        _, cluster_profiles = run_clustering(subset, cluster_features, n_clusters=n_clusters)

    insight_dict = generate_insight(subset, selected_kpi, selected_kpi_2,
                                stats=stats_for_msg, cluster_profiles=cluster_profiles)
    show_insight_cards(insight_dict)
# --------------------------
# 11) Unified Analysis Section (Dynamic by Method)
# --------------------------

st.markdown("## 🔍 Analysis Results")
st.caption("Select analysis type and KPIs from the sidebar to explore relationships, structures, or patterns in Luxembourg's demographic data.")

# --- Sidebar: Unified Analysis Settings ---
st.sidebar.markdown("---")
st.sidebar.header("📈 Analysis Settings")

analysis_method = st.sidebar.selectbox(
    "Select Analysis Method:",
    [
        "Correlation Matrix",
        "Principal Component Analysis (PCA)",
        "Z-score Profiling",
        "Composite Index"
    ],
    index=0
)

if analysis_method in ["Correlation Matrix", "Principal Component Analysis (PCA)", "Composite Index"]:
    selected_features = st.sidebar.multiselect(
        "Select KPIs for analysis:",
        options=list(kpi_options.keys()),
        default=[
            "share_65p",
            "share_foreign_citizenship",
            "share_not_in_family_nucleus",
            "old_dep_ratio"
        ],
        format_func=lambda x: kpi_options[x]
    )
elif analysis_method == "Z-score Profiling":
    selected_features = [st.sidebar.selectbox(
        "Select KPI for Z-score profiling:",
        options=list(kpi_options.keys()),
        format_func=lambda x: kpi_options[x]
    )]
else:
    selected_features = []

# =====================================================
#  📊 ANALYSIS BRANCHING
# =====================================================

# ---------- 1️⃣ Correlation Matrix ----------
if analysis_method == "Correlation Matrix":
    st.markdown("### 🔗 Correlation Matrix")
    make_correlation_heatmap(subset, selected_features)

# ---------- 2️⃣ PCA ----------
elif analysis_method == "Principal Component Analysis (PCA)":
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import plotly.express as px

    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least two KPIs for PCA.")
    else:
        st.markdown("### 🔬 Principal Component Analysis (PCA)")

        df_pca = subset[selected_features].dropna()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_pca)

        pca = PCA(n_components=2)
        components = pca.fit_transform(X_scaled)
        var_ratio = pca.explained_variance_ratio_

        pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
        pca_df["commune"] = subset.loc[df_pca.index, "commune"].values

        # --- クラスタリングが有効なら色分け ---
        color_col = "cluster" if enable_cluster and "cluster" in subset.columns else None

        fig_pca = px.scatter(
            pca_df,
            x="PC1",
            y="PC2",
            color=color_col,
            title=f"PCA Projection (PC1 {var_ratio[0]*100:.1f}%, PC2 {var_ratio[1]*100:.1f}%)",
            hover_name="commune",
            color_discrete_sequence=px.colors.qualitative.Set2 if color_col else ["#38bdf8"],
            opacity=0.75
        )
        fig_pca.update_traces(marker=dict(size=9, line=dict(width=0)))
        st.plotly_chart(fig_pca, use_container_width=True, key="pca_plot")

        # --- KPI寄与度（ロード）バー ---
        st.markdown("#### 🧭 KPI Contribution to PC1")
        loadings = pd.Series(np.abs(pca.components_[0]), index=selected_features).sort_values(ascending=True)
        fig_load = px.bar(
            loadings,
            x=loadings.values * 100,
            y=[kpi_options[k] for k in loadings.index],
            orientation="h",
            color=loadings.values,
            color_continuous_scale="Blues",
            height=450,
            title="Contribution of Each KPI to PC1 (%)"
        )
        fig_load.update_layout(
            xaxis_title="Loading Strength (%)",
            yaxis_title="",
            coloraxis_showscale=False,
            plot_bgcolor="#1e293b",
            paper_bgcolor="#1e293b",
            font_color="#e2e8f0"
        )
        st.plotly_chart(fig_load, use_container_width=True, key="pca_loadings")
        st.caption("""
        **Interpretation:**  
        PCA identifies hidden demographic dimensions across communes.  
        - **PC1 (horizontal)** → captures ageing vs. foreign population contrast.  
        - **PC2 (vertical)** → highlights isolation vs. family cohesion patterns.  
        Communes close together share similar socio-demographic structures.
        """)
# ---------- 3️⃣ Z-score Profiling ----------
elif analysis_method == "Z-score Profiling":
    import scipy.stats as stats
    kpi = selected_features[0]
    st.markdown(f"### 📈 Z-score Profiling – {kpi_options[kpi]}")

    df_z = subset[["commune", kpi]].dropna()
    df_z["Z-score"] = stats.zscore(df_z[kpi])

    fig_z = px.bar(
        df_z.sort_values("Z-score", ascending=False),
        x="commune",
        y="Z-score",
        title="Z-score by Commune",
        color="Z-score",
        color_continuous_scale="RdBu",
        range_color=(-3, 3),
        height=500
    )
    st.plotly_chart(fig_z, use_container_width=True)

    st.caption("Communes with |Z| > 2 are considered statistically significant outliers.")

    # 上位3 / 下位3表示
    st.markdown("#### 🏆 Top / Bottom Communes")
    top3 = df_z.nlargest(3, "Z-score")
    bot3 = df_z.nsmallest(3, "Z-score")

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Top 3 (High positive deviation)**")
        st.table(top3[["commune", "Z-score"]].set_index("commune"))
    with cols[1]:
        st.markdown("**Bottom 3 (Low negative deviation)**")
        st.table(bot3[["commune", "Z-score"]].set_index("commune"))

# ---------- 4️⃣ Composite Index ----------
elif analysis_method == "Composite Index":
    if len(selected_features) < 2:
        st.warning("⚠️ Please select at least two KPIs to build a composite index.")
    else:
        st.markdown("### 🧮 Composite Index – Demographic Stress Score")

        # Normalize and average (equal weight for now)
        df_idx = subset[["commune"] + selected_features].dropna().copy()
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_idx[selected_features])
        df_idx["Composite Index"] = X_scaled.mean(axis=1)

        fig_idx = px.choropleth_mapbox(
            df_idx,
            geojson=json.loads(geojson_str),
            locations=df_idx.index,
            color="Composite Index",
            hover_name="commune",
            mapbox_style="carto-positron",
            color_continuous_scale="Plasma",
            opacity=0.8,
            height=600
        )
        fig_idx.update_layout(
            mapbox_center={"lat": 49.815, "lon": 6.13},
            mapbox_zoom=8,
            margin={"r":0,"t":0,"l":0,"b":0}
        )
        st.plotly_chart(fig_idx, use_container_width=True)

        # 上位/下位リスト
        st.markdown("#### 🧭 Top / Bottom Communes by Composite Index")
        top5 = df_idx.nlargest(5, "Composite Index")[["commune", "Composite Index"]]
        bot5 = df_idx.nsmallest(5, "Composite Index")[["commune", "Composite Index"]]

        cols = st.columns(2)
        with cols[0]:
            st.markdown("**Top 5 Communes (Highest Stress)**")
            st.table(top5.set_index("commune"))
        with cols[1]:
            st.markdown("**Bottom 5 Communes (Lowest Stress)**")
            st.table(bot5.set_index("commune"))
# --------------------------
# 12) Footer
# --------------------------
st.markdown("---")
st.caption("Source: STATEC / LUSTAT Census | Visualization by Keiji Uehara – Enhanced version")