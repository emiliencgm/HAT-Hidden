from typing import Optional
import warnings

from fastcluster import complete as max_linkage
import numpy as np
from numpy.typing import ArrayLike
from scipy.cluster.hierarchy import fcluster
from scipy.integrate import trapezoid
from sklearn.preprocessing import MinMaxScaler

from rogi.rogi_xd_utils import IntegrationDomain, distance_matrix



def nmoment(x: ArrayLike, center: Optional[float] = None, n: int = 2, weights: Optional[ArrayLike] = None) -> float:
    """calculate the n-th moment with given sample weights

    Parameters
    ----------
    x : array_like
        samples
    c : Optional[float], default=None
        central location. If None, the average of `x` is used
    n : int, default=2
        the moment to calculate
    w : Optional[ArrayLike], default=None
        sample weights, if weighted moment is to be computed.

    Returns
    -------
    float
        the n-th moment
    """
    x = np.array(x)
    center = center if center is not None else np.average(x, weights=weights)

    return np.average((x - center) ** n, weights=weights)

def coarsened_sd(y: np.ndarray, Z: np.ndarray, t: float) -> float:
    """the coarsened standard deviation of the samples `y`

    The coarsened moment is calculated via clustering the input samples `y` according to the input
    linkage matrix `Z` and distance threhsold `t` and calculating the mean value of each cluster.
    The 2nd weighted moment (variance) of these cluster means is calculated with weights equal to
    the size of the respective cluster.

    NOTE: The samples are assumed to lie in the range [0, 1], so the coarsened standard deviation
    is multiplied by 2 to normalize it to the range [0, 1].

    Parameters
    ----------
    y : np.ndarray
        the original samples
    Z : np.ndarray
        the linkage matrix from hierarchical cluster. See :func:`scipy.cluster.hierarchy.linkage`
        for more details
    t : float
        the distance threshold to apply when forming clusters

    Returns
    -------
    float
        the coarsened standard deviation
    """
    if (y < 0).any() or (y > 1).any():
        warnings.warn("Input array 'y' has values outside of [0, 1]")

    cluster_ids = fcluster(Z, t, "distance") #return T=[1,1,1,2,2,3,3,...,n_cluster,n_cluster], where T[i] represents the clustering label of i-th data point

    clusters = set(cluster_ids)

    means = []
    weights = []
    for i in clusters:
        mask = cluster_ids == i
        means.append(y[mask].mean())
        weights.append(len(y[mask]))

    # max std dev is 0.5 --> multiply by 2 so that results is in [0,1]
    var = nmoment(means, n=2, weights=weights)
    sd_normalized = 2 * np.sqrt(var)


    return sd_normalized, len(clusters)

def coarse_grain(D: np.ndarray, y: np.ndarray, min_dt: float = 0.01):
    # print('Clustering...')
    Z = max_linkage(D)
    all_distance_thresholds = Z[:, 2]

    # print(f"Subsampling with minimum step size of {min_dt:0.3f}")
    thresholds = []
    t_prev = -1
    for t in all_distance_thresholds:
        if t < t_prev + min_dt:
            continue

        thresholds.append(t)
        t_prev = t

    # print(f"Coarsening with thresholds {flist(thresholds):0.3f}")
    
    cg_sds, n_clusters = zip(*[coarsened_sd(y, Z, t) for t in thresholds])

    # when num_clusters == num_data ==> stddev/skewness of dataset
    thresholds = np.array([0.0, *thresholds, 1.0])
    cg_sds = np.array([2 * y.std(), *cg_sds, 0])
    n_clusters = np.array([len(y), *n_clusters, 1])
    
    return thresholds, cg_sds, n_clusters

def rogi(x, y, normalize=True, min_dt=0.01, domain: IntegrationDomain = IntegrationDomain.LOG_CLUSTER_RATIO):
    """
    x: np.ndarray, raw features of all data points. x is then transformed into distance matrix (only the upper triangle is used) and **standardized** by dividing max(Dx).
    
    normalize : whether to normalize the property values to the range [0, 1].

    min_dt : the mimimum distance to use between threshold values when coarse graining the dataset.
    """

    if normalize:
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1))[:, 0]
    elif (y < 0).any() or (y > 1).any():
        warnings.warn("Input array 'y' has values outside [0, 1]. ROGI may be outside [0, 1]!")

    Dx = distance_matrix(X=x)
    
    thresholds, cg_sds, n_clusters = coarse_grain(Dx, y, min_dt)
        
    if domain == IntegrationDomain.THRESHOLD:
        x = thresholds
    elif domain == IntegrationDomain.CLUSTER_RATIO:
        x = 1 - n_clusters / n_clusters[0]
    else:
        x = 1 - np.log(n_clusters) / np.log(n_clusters[0])

    score: float = cg_sds[0] - trapezoid(cg_sds, x)

    return score