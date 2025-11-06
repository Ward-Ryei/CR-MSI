import umap
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA,NMF,FactorAnalysis,KernelPCA,TruncatedSVD,FastICA,SparsePCA
def dimension_reduction_UMAP(X,n_components=10):
    return umap.UMAP(n_components=n_components).fit_transform(X)
def dimension_reduction_TSNE(X,n_components=10):
    return TSNE(n_components=2).fit_transform(X)
def dimension_reduction_PCA(X,n_components=10):
    return PCA(n_components=n_components).fit_transform(X)
def dimension_reduction_NMF(X,n_components=10):
    return NMF(n_components=n_components).fit_transform(X)
def dimension_reduction_FA(X,n_components=10):
    return FactorAnalysis(n_components=n_components).fit_transform(X)
def dimension_reduction_KPCA(X,n_components=10):
    return KernelPCA(n_components=n_components).fit_transform(X)
def dimension_reduction_TSVD(X,n_components=10):
    return TruncatedSVD(n_components=n_components).fit_transform(X)
def dimension_reduction_FICA(X,n_components=10):
    return FastICA(n_components=n_components).fit_transform(X)
def dimension_reduction_SPCA(X,n_components=10):
    return SparsePCA(n_components=n_components).fit_transform(X)
    
    
    
DR_FUNC={"UMAP":dimension_reduction_UMAP,"TSNE":dimension_reduction_TSNE,
    "PCA":dimension_reduction_PCA,"NMF":dimension_reduction_NMF,
    "FA":dimension_reduction_FA,"KPCA":dimension_reduction_KPCA,
    "TSVD":dimension_reduction_TSVD,"FICA":dimension_reduction_FICA,
    "SPCA":dimension_reduction_SPCA,
    }