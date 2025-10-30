# 🏙️ Luxembourg Ageing Dashboard – STATEC Hackathon 2025

An interactive **Streamlit dashboard** developed for the **STATEC Hackathon 2025**,  
analyzing and visualizing **ageing and demographic patterns across communes in Luxembourg (2021)**.

This project was designed and implemented by **Keiji Uehara**, combining data visualization, clustering, and exploratory statistical analysis to reveal socio-demographic structures within Luxembourg.

---

## 📊 Overview

The **Luxembourg Ageing Dashboard** provides an accessible interface for exploring ageing-related indicators at the commune level.

Users can:
- Visualize key demographic ratios and their spatial distribution.  
- Compare correlations between multiple indicators.  
- Perform clustering to identify commune profiles.  
- Explore statistical summaries through PCA, Z-score profiling, and composite indices.

### ✳️ Core KPIs
| Code | Indicator |
|------|------------|
| `share_65p` | Share of population aged 65+ (%) |
| `share_foreign_citizenship` | Share of foreign citizens (%) |
| `share_not_in_family_nucleus` | Share not in a family nucleus (%) |
| `old_dep_ratio` | Old-age dependency ratio (%) |
| `youth_dep_ratio` | Youth dependency ratio (%) |
| `elderly_isolation_index` | Elderly isolation index (65+ × non-family %) |

---

## Key Features

### 🗺️ 1. Interactive Map Visualization
- Commune-level choropleth map built with **Plotly + Mapbox**  
- Dynamic selection of primary KPI for color scaling  
- Hover tooltips displaying detailed metrics  

### 🔍 2. Comparative Analysis
- Bivariate scatter plots and optional linear regression  
- Automatic computation of slope and R² values (via `statsmodels`)  

### 🧩 3. Commune Clustering
- K-Means clustering with selectable features  
- Auto-generated cluster labels (e.g., *Aged & Isolated*, *Diverse & Younger*)  
- Cluster summary table with color-coded profiles  

### 📈 4. Advanced Analysis Tools
- **Correlation Matrix** (Pearson): identify relationships among KPIs  
- **Principal Component Analysis (PCA)**: visualize underlying dimensions  
- **Z-score Profiling**: detect statistical outliers among communes  
- **Composite Index**: create a combined “demographic stress score”  

---

## Tech Stack

| Category | Tools & Libraries |
|-----------|-------------------|
| Framework | Streamlit |
| Data Manipulation | pandas, geopandas |
| Visualization | Plotly (Express & Graph Objects) |
| Statistics | statsmodels, scipy |
| Machine Learning | scikit-learn |
| Language | Python 3.11 |

---

## Repository Structure
statec_hackathon_2025/
├── app/
│   ├── app.py              # Main Streamlit app
│── outputs/                # Processed GeoJSON data and derived KPIs
├── data/                   # raw input files before preprocessing
├── README.md               # Project documentation
├── requirements.txt        # Dependencies
└── LICENSE

---

## How to Run Locally

```bash
# Clone the repository
git clone https://github.com/keijiingermany/statec_hackathon_2025.git
cd statec_hackathon_2025/app

# (Optional) Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

Then open your browser at:
	http://localhost:8501

⸻

Author

Keiji Uehara
	•	Data preparation, analysis, visualization, and full Streamlit app development
	•	GitHub: keijiingermany
	•	Contact: Available upon request (Hackathon submission)

⸻

Methodological Notes

The analysis is based on commune-level data (2021) derived from Luxembourg’s national census.
Indicators were standardized and combined through statistical transformations (Z-score, PCA)
to identify latent demographic structures.

Clustering Approach
	•	Algorithm: K-Means (scikit-learn)
	•	Standardized Inputs: share_65p, share_foreign_citizenship, share_not_in_family_nucleus
	•	Number of clusters: 2–6 (selectable by user)
	•	Summary Table: Automatically labeled clusters with color-coded profiles

Analytical Extensions (for future releases)
	•	Multi-year data ingestion (2021–2030, expandable via GeoJSON updates)
	•	Time-series comparison across communes
	•	Predictive modelling of ageing and isolation risks

⸻

Data Source

All datasets and indicators used in this project originate exclusively from:

STATEC Open Data Portal
National Institute of Statistics and Economic Studies of Luxembourg
	https://lustat.statec.lu/

No external or private data sources were used.
All calculations, visualizations, and derived indicators are based solely on STATEC’s public datasets.

⸻

🏁 License

This project is shared for educational and non-commercial purposes
within the scope of the STATEC Hackathon 2025.
All underlying data remains © STATEC Luxembourg.
