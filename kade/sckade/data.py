from collections import defaultdict
from typing import List, Mapping, Optional

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse
import scipy.stats
import sklearn.cluster
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.neighbors
import sklearn.preprocessing
import sklearn.utils.extmath
from anndata import AnnData
from sklearn.preprocessing import normalize
from sparse import COO

from . import num
from .typehint import Kws
from .utils import logged


def lsi(
        adata: AnnData, n_components: int = 20,
        use_highly_variable: Optional[bool] = None, **kwargs
) -> None:

    if "random_state" not in kwargs:
        kwargs["random_state"] = 0  # Keep deterministic as the default behavior
    if use_highly_variable is None:
        use_highly_variable = "highly_variable" in adata.var
    adata_use = adata[:, adata.var["highly_variable"]] if use_highly_variable else adata
    X = num.tfidf(adata_use.X)
    X_norm = normalize(X, norm="l1")
    X_norm = np.log1p(X_norm * 1e4)
    X_lsi = sklearn.utils.extmath.randomized_svd(X_norm, n_components, **kwargs)[0]
    X_lsi -= X_lsi.mean(axis=1, keepdims=True)
    X_lsi /= X_lsi.std(axis=1, ddof=1, keepdims=True)
    adata.obsm["X_lsi"] = X_lsi


def aggregate_obs(
        adata: AnnData, by: str, X_agg: Optional[str] = "sum",
        obs_agg: Optional[Mapping[str, str]] = None,
        obsm_agg: Optional[Mapping[str, str]] = None,
        layers_agg: Optional[Mapping[str, str]] = None
) -> AnnData:
    
    obs_agg = obs_agg or {}
    obsm_agg = obsm_agg or {}
    layers_agg = layers_agg or {}

    by = adata.obs[by]
    agg_idx = pd.Index(by.cat.categories) \
        if pd.api.types.is_categorical_dtype(by) \
        else pd.Index(np.unique(by))
    agg_sum = scipy.sparse.coo_matrix((
        np.ones(adata.shape[0]), (
            agg_idx.get_indexer(by),
            np.arange(adata.shape[0])
        )
    )).tocsr()
    agg_mean = agg_sum.multiply(1 / agg_sum.sum(axis=1))

    agg_method = {
        "sum": lambda x: agg_sum @ x,
        "mean": lambda x: agg_mean @ x,
        "majority": lambda x: pd.crosstab(by, x).idxmax(axis=1).loc[agg_idx].to_numpy()
    }

    X = agg_method[X_agg](adata.X) if X_agg and adata.X is not None else None
    obs = pd.DataFrame({
        k: agg_method[v](adata.obs[k])
        for k, v in obs_agg.items()
    }, index=agg_idx.astype(str))
    obsm = {
        k: agg_method[v](adata.obsm[k])
        for k, v in obsm_agg.items()
    }
    layers = {
        k: agg_method[v](adata.layers[k])
        for k, v in layers_agg.items()
    }
    for c in obs:
        if pd.api.types.is_categorical_dtype(adata.obs[c]):
            obs[c] = pd.Categorical(obs[c], categories=adata.obs[c].cat.categories)
    return AnnData(
        X=X, obs=obs, var=adata.var,
        obsm=obsm, varm=adata.varm, layers=layers,
        dtype=None if X is None else X.dtype
    )


def transfer_labels(
        ref: AnnData, query: AnnData, field: str,
        n_neighbors: int = 30, use_rep: Optional[str] = None,
        key_added: Optional[str] = None, **kwargs
) -> None:

    xrep = ref.obsm[use_rep] if use_rep else ref.X
    yrep = query.obsm[use_rep] if use_rep else query.X
    xnn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors, **kwargs
    ).fit(xrep)
    ynn = sklearn.neighbors.NearestNeighbors(
        n_neighbors=n_neighbors, **kwargs
    ).fit(yrep)
    xx = xnn.kneighbors_graph(xrep)
    xy = ynn.kneighbors_graph(xrep)
    yx = xnn.kneighbors_graph(yrep)
    yy = ynn.kneighbors_graph(yrep)
    jaccard = (xx @ yx.T) + (xy @ yy.T)
    jaccard.data /= 4 * n_neighbors - jaccard.data
    normalized_jaccard = jaccard.multiply(1 / jaccard.sum(axis=0))
    onehot = sklearn.preprocessing.OneHotEncoder()
    xtab = onehot.fit_transform(ref.obs[[field]])
    ytab = normalized_jaccard.T @ xtab
    pred = pd.Series(
        onehot.categories_[0][ytab.argmax(axis=1).A1],
        index=query.obs_names, dtype=ref.obs[field].dtype
    )
    conf = pd.Series(
        ytab.max(axis=1).toarray().ravel(),
        index=query.obs_names
    )
    key_added = key_added or field
    query.obs[key_added] = pred
    query.obs[key_added + "_confidence"] = conf


@logged
def estimate_balancing_weight(
        *adatas: AnnData, use_rep: str = None, use_batch: Optional[str] = None,
        resolution: float = 1.0, cutoff: float = 0.5, power: float = 4.0,
        key_added: str = "balancing_weight"
) -> None:

    if use_batch:  # Recurse per batch
        estimate_balancing_weight.logger.info("Splitting batches...")
        adatas_per_batch = defaultdict(list)
        for adata in adatas:
            groupby = adata.obs.groupby(use_batch, dropna=False)
            for b, idx in groupby.indices.items():
                adata_sub = adata[idx]
                adatas_per_batch[b].append(AnnData(
                    obs=adata_sub.obs,
                    obsm={use_rep: adata_sub.obsm[use_rep]}
                ))
        if len(set(len(items) for items in adatas_per_batch.values())) != 1:
            raise ValueError("Batches must match across datasets!")
        for b, items in adatas_per_batch.items():
            estimate_balancing_weight.logger.info("Processing batch %s...", b)
            estimate_balancing_weight(
                *items, use_rep=use_rep, use_batch=None,
                resolution=resolution, cutoff=cutoff,
                power=power, key_added=key_added
            )
        estimate_balancing_weight.logger.info("Collating batches...")
        collates = [
            pd.concat([item.obs[key_added] for item in items])
            for items in zip(*adatas_per_batch.values())
        ]
        for adata, collate in zip(adatas, collates):
            adata.obs[key_added] = collate.loc[adata.obs_names]
        return

    if use_rep is None:
        raise ValueError("Missing required argument `use_rep`!")
    adatas_ = [
        AnnData(
            obs=adata.obs.copy(deep=False).assign(n=1),
            obsm={use_rep: adata.obsm[use_rep]}
        ) for adata in adatas
    ]  # Avoid unwanted updates to the input objects

    estimate_balancing_weight.logger.info("Clustering cells...")
    for adata_ in adatas_:
        sc.pp.neighbors(
            adata_, n_pcs=adata_.obsm[use_rep].shape[1],
            use_rep=use_rep, metric="cosine"
        )
        sc.tl.leiden(adata_, resolution=resolution)

    leidens = [
        aggregate_obs(
            adata, by="leiden", X_agg=None,
            obs_agg={"n": "sum"}, obsm_agg={use_rep: "mean"}
        ) for adata in adatas_
    ]
    us = [normalize(leiden.obsm[use_rep], norm="l2") for leiden in leidens]
    ns = [leiden.obs["n"] for leiden in leidens]

    estimate_balancing_weight.logger.info("Matching clusters...")
    cosines = []
    for i, ui in enumerate(us):
        for j, uj in enumerate(us[i + 1:], start=i + 1):
            cosine = ui @ uj.T
            cosine[cosine < cutoff] = 0
            cosine = COO.from_numpy(cosine)
            cosine = np.power(cosine, power)
            key = tuple(
                slice(None) if k in (i, j) else np.newaxis
                for k in range(len(us))
            )  # To align axes
            cosines.append(cosine[key])
    joint_cosine = num.prod(cosines)
    estimate_balancing_weight.logger.info(
        "Matching array shape = %s...", str(joint_cosine.shape)
    )

    estimate_balancing_weight.logger.info("Estimating balancing weight...")
    for i, (adata, adata_, leiden, n) in enumerate(zip(adatas, adatas_, leidens, ns)):
        balancing = joint_cosine.sum(axis=tuple(
            k for k in range(joint_cosine.ndim) if k != i
        )).todense() / n
        balancing = pd.Series(balancing, index=leiden.obs_names)
        balancing = balancing.loc[adata_.obs["leiden"]].to_numpy()
        balancing /= balancing.sum() / balancing.size
        adata.obs[key_added] = balancing


@logged
def get_metacells(
        *adatas: AnnData, use_rep: str = None, n_meta: int = None,
        common: bool = True, seed: int = 0,
        agg_kws: Optional[List[Kws]] = None
) -> List[AnnData]:

    if use_rep is None:
        raise ValueError("Missing required argument `use_rep`!")
    if n_meta is None:
        raise ValueError("Missing required argument `n_meta`!")
    adatas = [
        AnnData(
            X=adata.X,
            obs=adata.obs.set_index(adata.obs_names + f"-{i}"), var=adata.var,
            obsm=adata.obsm, varm=adata.varm, layers=adata.layers,
            dtype=None if adata.X is None else adata.X.dtype
        ) for i, adata in enumerate(adatas)
    ]  # Avoid unwanted updates to the input objects

    get_metacells.logger.info("Clustering metacells...")
    combined = ad.concat(adatas)
    try:
        import faiss
        kmeans = faiss.Kmeans(
            combined.obsm[use_rep].shape[1], n_meta,
            gpu=False, seed=seed
        )
        kmeans.train(combined.obsm[use_rep])
        _, combined.obs["metacell"] = kmeans.index.search(combined.obsm[use_rep], 1)
    except ImportError:
        get_metacells.logger.warning(
            "`faiss` is not installed, using `sklearn` instead... "
            "This might be slow with a large number of cells. "
            "Consider installing `faiss` following the guide from "
            "https://github.com/facebookresearch/faiss/blob/main/INSTALL.md"
        )
        kmeans = sklearn.cluster.KMeans(n_clusters=n_meta, random_state=seed)
        combined.obs["metacell"] = kmeans.fit_predict(combined.obsm[use_rep])
    for adata in adatas:
        adata.obs["metacell"] = combined[adata.obs_names].obs["metacell"]

    get_metacells.logger.info("Aggregating metacells...")
    agg_kws = agg_kws or [{}] * len(adatas)
    if not len(agg_kws) == len(adatas):
        raise ValueError("Length of `agg_kws` must match the number of datasets!")
    adatas = [
        aggregate_obs(adata, "metacell", **kws)
        for adata, kws in zip(adatas, agg_kws)
    ]
    if common:
        common_metacells = list(set.intersection(*(
            set(adata.obs_names) for adata in adatas
        )))
        if len(common_metacells) == 0:
            raise RuntimeError("No common metacells found!")
        return [adata[common_metacells].copy() for adata in adatas]
    return adatas