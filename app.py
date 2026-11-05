import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import *

st.title("Unsupervised models")

st.subheader("Dataframe")
file = st.file_uploader("Upload CSV", type=["csv"])

if file is None:
    st.info("Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(file)
st.dataframe(df, width='content')

st.subheader("Preprocessing")
features = st.multiselect("Choose features set", df.columns.tolist())

if not features:
    st.stop()

# Note: Modified call to include feature_drop column if needed, or default it
X = preprocessing(df, features, 'CustomerID' if 'CustomerID' in df.columns else df.columns[0])
st.dataframe(X, width='content')

st.subheader("Modeling")
model = st.selectbox("Choose one", ["KMeans", "KMedoids", "DBSCAN", "Agglomerative"])

labels = None

if model == "KMeans":
    k = st.slider("Value of K (n of clusters)", 2, 10, value=6)
    labels = kmeans(X, k)

elif model == "KMedoids":
    k = st.slider("Value of K (n of clusters)", 2, 10, value=6)
    metric = st.selectbox("Choose metric", ["manhattan", "euclidean"])
    labels = kmedoids(X, k, metric)

elif model == "DBSCAN":
    min_samples = st.slider("Select min samples", 1, 100, value=2)
    eps = st.number_input("Epsilon (e.g 0.5)", min_value=0.01, max_value=100.0, value=0.1)
    labels = dbscan(X, eps, min_samples)

elif model == "Agglomerative":
    k = st.slider("Value of K (n of clusters)", 2, 10, value=6)
    linkage = st.selectbox("Choose one linkage", ["ward", "complete", "average", "single"])
    labels = hierarchicalClustering(X, k, linkage)

if labels is not None:
    # Calculate metrics
    try:
        sil, dav = metrics(X, labels)
        st.metric("Silhouette Score", f"{sil:.3f}")
        st.metric("Davies-Bouldin Index", f"{dav:.3f}")
    except Exception as e:
        st.warning(f"Could not calculate metrics: {e}")

    st.subheader("Plotting of two variables")
    plot_features = st.multiselect("Choose features to plot (2 max.)", X.columns.tolist())

    if len(plot_features) < 2:
        st.info("Choose two features to begin plot")
    elif len(plot_features) == 2:
        fig, ax = plt.subplots()
        scatter = ax.scatter(X[plot_features[0]], X[plot_features[1]], c=labels, cmap='viridis')
        ax.set_xlabel(plot_features[0])
        ax.set_ylabel(plot_features[1])
        st.pyplot(fig)
    elif len(plot_features) > 2:
        st.error("More than two features. Impossible to plot")