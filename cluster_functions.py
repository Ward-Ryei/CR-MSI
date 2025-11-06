import os
os.environ["OMP_NUM_THREADS"] = "12"
from sklearn.cluster import KMeans,DBSCAN,Birch,AffinityPropagation,AgglomerativeClustering,FeatureAgglomeration,MeanShift,OPTICS,SpectralClustering,SpectralBiclustering,SpectralCoclustering
#from sklearn.cluster import HDBSCAN
from sklearn.mixture import GaussianMixture



def cluster_by_kmeans(X,n_cluster=10,*args):
    kmeans=KMeans(init='k-means++',n_clusters=n_cluster,n_init=10,random_state=1)#int(time.time())
    kmeans.fit(X)
    return kmeans.labels_
#DBSCAN
def cluster_by_DBSCAN(X,*args):
    dbscan = DBSCAN()
    dbscan.fit(X)
    return dbscan.labels_

def cluster_by_GMM(X,n_clusters=10,*args):
    gmm=GaussianMixture(n_components=n_clusters)
    lables=gmm.fit_predict(X)
    return lables
def cluster_by_Birch(X,n_clusters=10,*args):
    return Birch(n_clusters=n_clusters).fit_predict(X)


def cluster_by_AgglomerativeClustering(X,n_clusters=10,*args):
    return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(X)


def cluster_by_MeanShift(X,n_clusters=10,*args):
    return MeanShift().fit_predict(X)

def cluster_by_OPTICS(X,n_clusters=10,*args):
    return OPTICS().fit_predict(X)

def cluster_by_SpectralClustering(X,n_clusters=10,*args):
    return SpectralClustering(n_clusters=n_clusters).fit_predict(X)

def cluster_by_SpectralBiclustering(X,n_clusters=10,*args):
    model= SpectralBiclustering(n_clusters=n_clusters)
    model.fit(X)
    return model.row_labels_

def cluster_by_SpectralCoclustering(X,n_clusters=10,*args):
    model= SpectralCoclustering(n_clusters=n_clusters)
    model.fit(X)
    return model.row_labels_

CLUSTER_FUNC={"Kmeans":cluster_by_kmeans,"GMM":cluster_by_GMM,"DBSCAN":cluster_by_DBSCAN,
    "Birch":cluster_by_Birch,"Agglo":cluster_by_AgglomerativeClustering,
    "MeanShift":cluster_by_MeanShift,"OPTICS":cluster_by_OPTICS,
    "SpClu":cluster_by_SpectralClustering,"SpBic":cluster_by_SpectralBiclustering,
    "SpCoc":cluster_by_SpectralCoclustering}