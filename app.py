import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from model import *

st.title("Unsupervised Models App")

# 1. Upload CSV & Preview
file = st.file_uploader("Upload CSV file", type=["csv"])
if file is None:
    st.info("Please upload a CSV file to start.")
    st.stop()

df = pd.read_csv(file)
st.subheader("Dataset Preview")
st.dataframe(df.head())

# 2. Feature Selection
features = st.multiselect("Select features for modeling", df.columns.tolist(), default=df.columns.tolist()[1:])
if len(features) < 2:
    st.warning("Please select at least 2 features.")
    st.stop()

# Base Preprocessing - using (df, features, dummy_drop_col) or just passing CustomerID as a drop if available
drop_candidate = 'CustomerID' if 'CustomerID' in df.columns and 'CustomerID' not in features else df.columns[0]
X_base = preprocessing(df, features)

# Feature Selection Mode (Manual vs PCA Reduction)
st.subheader("Feature Selection Mode")
mode = st.radio("Choose Mode", ["Manual Selection", "PCA Reduction"])

if mode == "Manual Selection":
    X = X_base.copy()
else:
    max_components = min(X_base.shape[0], X_base.shape[1])
    if max_components < 2:
        st.error("Not enough dimensions to perform PCA (minimum 2 required).")
        st.stop()
    n_components = st.slider("PCA Components", 2, max_components, 2)
    pca_model = PCA(n_components=n_components)
    X = pd.DataFrame(pca_model.fit_transform(X_base), columns=[f"PC{i+1}" for i in range(n_components)], index=X_base.index)
    st.write(f"PCA active. Retained variance: {pca_model.explained_variance_ratio_.sum():.2%}")

# 3. Model Selection & Hyperparameters
st.subheader("Clustering Configuration")
model_name = st.selectbox("Select Algorithm", ["K-Means", "K-Medoids", "DBSCAN", "Hierarchical"])

labels = None
if model_name == "K-Means":
    k = st.slider("Number of clusters (k)", 2, 10, 6)
    labels = kmeans(X, k)

elif model_name == "K-Medoids":
    k = st.slider("Number of clusters (k)", 2, 10, 6)
    metric = st.selectbox("Distance Metric", ["manhattan", "euclidean"])
    labels = kmedoids(X, k, metric)

elif model_name == "DBSCAN":
    eps = st.number_input("Epsilon (eps)", 0.01, 10.0, 0.5)
    min_samples = st.slider("Minimum Samples", 1, 50, 5)
    labels = dbscan(X, eps, min_samples)

elif model_name == "Hierarchical":
    k = st.slider("Number of clusters (k)", 2, 10, 6)
    linkage = st.selectbox("Linkage Criterion", ["ward", "complete", "average", "single"])
    labels = hierarchicalClustering(X, k, linkage)

# 4. Display Clustered Dataset & Metrics
if labels is not None:
    df['Cluster'] = labels
    st.subheader("Clustered Dataset")
    st.dataframe(df)

    # Metrics
    try:
        sil, db = metrics(X, labels)
        col1, col2 = st.columns(2)
        col1.metric("Silhouette Score", f"{sil:.4f}")
        col2.metric("Davies-Bouldin Index", f"{db:.4f}")
    except Exception as e:
        st.error(f"Could not calculate metrics: {e}")

    # 5. Visualization
    st.subheader("Cluster Visualization")
    fig, ax = plt.subplots()
    if X.shape[1] == 2:
        ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels, cmap='viridis', edgecolors='k')
        ax.set_xlabel(X.columns[0])
        ax.set_ylabel(X.columns[1])
    else:
        # Automatically apply PCA to reduce dimensions to 2D for visualization if > 2 features
        pca_coords = PCA(n_components=2).fit_transform(X)
        ax.scatter(pca_coords[:, 0], pca_coords[:, 1], c=labels, cmap='viridis', edgecolors='k')
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_title("PCA Projected Cluster Space")
    st.pyplot(fig)