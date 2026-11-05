import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model import preprocessing, clustering, metrics

st.title("Unsupervised learning app")

st.subheader("Dataframe")
file = st.file_uploader("Upload CSV", type=["csv"])

if file is None:
    st.info("Upload a CSV file to begin")
    st.stop()

df = pd.read_csv(file)
st.dataframe(df, width='content')

st.subheader("Preprocessing")
features = st.multiselect("Choose features set", df.columns.tolist())

if features == []:
    st.stop()

X = preprocessing(df, features)
st.dataframe(X, width='content')

st.subheader("Modeling")
k = st.slider("Value of K (n of clusters)", 2, 10)
linkage = st.selectbox("Choose one linkage", ["ward", "complete", "average", "single"])

labels = clustering(X, k, linkage)
sil, dav = metrics(X, labels)

st.metric("Silhouette Score", f"{sil:.3f}")
st.metric("Davies-Bouldin Index", f"{dav:.3f}")

st.subheader("Plotting of two variables")

plot_features = st.multiselect("Choose features to plot (2 max.)", X.columns.tolist())

if len(plot_features) < 2:
    st.info("Choose two features to begin plot")
elif len(plot_features) == 2:
    fig, ax = plt.subplots()
    ax.scatter(X[plot_features[0]], X[plot_features[1]], c=labels)
    ax.set_xlabel(plot_features[0])
    ax.set_ylabel(plot_features[1])
    st.pyplot(fig)
elif len(plot_features) > 2:
    st.error("More than two features. Imposible to plot")
else:
    st.error("An error happened")