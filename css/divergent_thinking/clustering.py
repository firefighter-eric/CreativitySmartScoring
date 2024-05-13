from sklearn.cluster import AgglomerativeClustering


def get_cluster(x):
    clustering = AgglomerativeClustering(n_clusters=10).fit(x)
    return clustering
