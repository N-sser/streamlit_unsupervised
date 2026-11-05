import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
#!pip install scikit-learn-extra
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score, davies_bouldin_score

def preprocessing(df, features: list, feature_drop: str):
    df = df.drop(columns=feature_drop, axis=1)
    X = df[features].copy()
    numerical = df.select_dtypes(include=['int64', 'float64']).columns
    categorical = df.select_dtypes(include=['object']).columns

    ss = StandardScaler()
    X[numerical] = ss.fit_transform(X[numerical])

    le = LabelEncoder()
    X[categorical] = X[categorical].apply(le.fit_transform)
    return X

def kmeans(X, n_clusters):
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    y_kmeans = kmeans.fit_predict(X)
    return y_kmeans

def kmedoids(X, n_clusters, metric):
    kmedoids = KMedoids(n_clusters=n_clusters, metric=metric)
    y_kmedoids = kmedoids.fit_predict(X)
    return y_kmedoids

def dbscan(X, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    y_dbscan = dbscan.fit_predict(X)
    return y_dbscan

def hierarchicalClustering(X, n_clusters, linkage):
    hc = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
    y_hc = hc.fit_predict(X)
    return y_hc

def metrics(X, y_hc):
    return silhouette_score(X, y_hc), davies_bouldin_score(X, y_hc)