import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium
from folium.plugins import HeatMap

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import linkage, fcluster

# --------------------------------------------------
# App Config
# --------------------------------------------------
st.set_page_config(page_title="Crime Clustering Intelligence System", layout="wide")
st.title("🚨 Crime Clustering Intelligence System")

# --------------------------------------------------
# Utility Functions
# --------------------------------------------------
@st.cache_data
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        return pd.read_csv(file, delimiter=None, engine="python")

def evaluate_clustering(X, labels, name):
    if len(set(labels)) <= 1 or -1 in labels:
        return {
            "Model": name,
            "Silhouette": -1,
            "Davies-Bouldin": np.inf,
            "Calinski-Harabasz": 0,
            "Valid": False
        }
    return {
        "Model": name,
        "Silhouette": silhouette_score(X, labels),
        "Davies-Bouldin": davies_bouldin_score(X, labels),
        "Calinski-Harabasz": calinski_harabasz_score(X, labels),
        "Valid": True
    }

def plot_clusters(X_pca, labels, title):
    fig, ax = plt.subplots()
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10", s=25)
    ax.set_title(title)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    st.pyplot(fig)

def cluster_explainability(df, features, labels, idx):
    df_exp = df.loc[idx].copy()
    df_exp["Cluster"] = labels
    overall_mean = df_exp[features].mean()

    summary = {}
    for c in sorted(df_exp["Cluster"].unique()):
        cdf = df_exp[df_exp["Cluster"] == c]
        diff = (cdf[features].mean() - overall_mean).abs().sort_values(ascending=False)
        summary[c] = {
            "size": len(cdf),
            "top_features": diff.head(3)
        }
    return summary

# --------------------------------------------------
# File Upload
# --------------------------------------------------
file = st.file_uploader("Upload dataset (.csv, .xlsx, .txt)", type=["csv", "xlsx", "txt"])

if not file:
    st.info("Please upload a dataset to begin.")
    st.stop()

df = load_data(file)
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

# --------------------------------------------------
# Time Feature Engineering
# --------------------------------------------------
for col in df.columns:
    try:
        df[col] = pd.to_datetime(df[col])
        df["Hour"] = df[col].dt.hour
        df["Month"] = df[col].dt.month
        break
    except:
        continue

# --------------------------------------------------
# Feature Selection
# --------------------------------------------------
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
if len(numeric_cols) < 2:
    st.error("At least 2 numeric columns are required.")
    st.stop()

features = st.multiselect("Select features for clustering", numeric_cols, default=numeric_cols)
df_model = df[features].dropna()

# --------------------------------------------------
# Scaling & PCA
# --------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.header("🔧 Hyperparameter Tuning")

k = st.sidebar.slider("KMeans / Hierarchical clusters (k)", 2, 10, 4)
eps = st.sidebar.slider("DBSCAN eps", 0.1, 5.0, 0.5)
min_samples = st.sidebar.slider("DBSCAN min_samples", 3, 10, 5)

# --------------------------------------------------
# Clustering Models
# --------------------------------------------------
labels_km = KMeans(n_clusters=k, random_state=42).fit_predict(X_scaled)
labels_hc = fcluster(linkage(X_scaled, method="ward"), t=k, criterion="maxclust")
labels_db = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(X_scaled)

results = pd.DataFrame([
    evaluate_clustering(X_scaled, labels_km, "KMeans"),
    evaluate_clustering(X_scaled, labels_hc, "Hierarchical"),
    evaluate_clustering(X_scaled, labels_db, "DBSCAN")
])

# --------------------------------------------------
# Best Model Selection
# --------------------------------------------------
results["Rank"] = (
    results["Silhouette"].rank(ascending=False) +
    results["Calinski-Harabasz"].rank(ascending=False) +
    results["Davies-Bouldin"].rank()
)
best_model = results.sort_values("Rank").iloc[0]["Model"]

label_map = {
    "KMeans": labels_km,
    "Hierarchical": labels_hc,
    "DBSCAN": labels_db
}

# --------------------------------------------------
# Display Best Model
# --------------------------------------------------
st.markdown("## 🥇 Best Clustering Model")
plot_clusters(X_pca, label_map[best_model], f"{best_model} (Best)")
st.dataframe(results[results["Model"] == best_model])

st.success(
    f"{best_model} performs best due to superior cluster separation, compactness, "
    "and consistency based on combined evaluation metrics."
)

# --------------------------------------------------
# Explainability
# --------------------------------------------------
st.markdown("## 🔍 Cluster Explainability")
explain = cluster_explainability(df, features, label_map[best_model], df_model.index)

for c, info in explain.items():
    st.subheader(f"Cluster {c}")
    st.write(f"Size: {info['size']}")
    st.dataframe(info["top_features"].reset_index().rename(
        columns={"index": "Feature", 0: "Deviation"}
    ))

# --------------------------------------------------
# Time-Based Analysis
# --------------------------------------------------
if "Hour" in df.columns:
    st.markdown("## 🌙 Day vs Night Crime Patterns")
    df_time = df.loc[df_model.index].copy()
    df_time["Cluster"] = label_map[best_model]
    df_time["Period"] = np.where(df_time["Hour"].between(6, 18), "Day", "Night")
    st.dataframe(df_time.groupby(["Cluster", "Period"]).size().unstack(fill_value=0))

if "Month" in df.columns:
    st.markdown("## 📅 Seasonal Crime Trends")
    fig, ax = plt.subplots()
    df_time.groupby("Month").size().plot(kind="bar", ax=ax)
    st.pyplot(fig)

# --------------------------------------------------
# Geo-Spatial Intelligence
# --------------------------------------------------
lat = [c for c in df.columns if "lat" in c.lower()]
lon = [c for c in df.columns if "lon" in c.lower()]

if lat and lon:
    st.markdown("## 🗺️ Crime Hotspots")
    geo_df = df.loc[df_model.index].copy()
    geo_df["Cluster"] = label_map[best_model]

    m = folium.Map(
        location=[geo_df[lat[0]].mean(), geo_df[lon[0]].mean()],
        zoom_start=11
    )

    HeatMap(geo_df[[lat[0], lon[0]]].values, radius=10).add_to(m)
    st.components.v1.html(m._repr_html_(), height=500)

# --------------------------------------------------
# Automated Recommendations
# --------------------------------------------------
st.markdown("## 🤖 Automated Insights")

largest_cluster = max(explain, key=lambda x: explain[x]["size"])
st.success(f"Cluster {largest_cluster} represents the highest crime concentration.")

if "Hour" in df.columns and (df_time["Period"] == "Night").mean() > 0.5:
    st.success("High crime intensity detected during night hours.")

if lat and lon:
    st.success("Geospatial clustering reveals concentrated crime hotspots.")

# --------------------------------------------------
# Other Models Comparison
# --------------------------------------------------
st.markdown("## 🥈 Other Models")

for model in label_map:
    if model == best_model:
        continue
    st.subheader(model)
    plot_clusters(X_pca, label_map[model], model)
    st.dataframe(results[results["Model"] == model])
    st.info("Not selected due to inferior separation, noise sensitivity, or instability.")
