from sklearn.cluster import AgglomerativeClustering
import numpy as np


def get_cluster(x):
    clustering = AgglomerativeClustering(n_clusters=10).fit(x)
    return clustering
