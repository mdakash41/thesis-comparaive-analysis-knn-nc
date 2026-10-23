"""Structural measures used in Table 4.1 and the synthetic experiments."""

import warnings
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def scale_heterogeneity(X):
    """log10(max feature std / min feature std) on raw features. 0 = equal scales."""
    s = X.std(axis=0)
    s = s[s > 0]
    return np.log10(s.max() / s.min())


def _per_class_feature(X, y, fn):
    vals = []
    for c in np.unique(y):
        Xc = X[y == c]
        for f in range(Xc.shape[1]):
            col = Xc[:, f]
            if col.std() > 0:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)  # near-constant columns
                    vals.append(abs(fn(col)))
    vals = [v for v in vals if np.isfinite(v)]
    return np.mean(vals) if vals else np.nan


def skewness(X, y):
    """Mean |skewness| over features and classes, raw X."""
    return _per_class_feature(X, y, skew)


def excess_kurtosis(X, y):
    """Mean |excess kurtosis| over features and classes, raw X."""
    return _per_class_feature(X, y, kurtosis)


def separability(X, y):
    """Mean pairwise centroid distance / mean within-class std, standardised X."""
    Xs = StandardScaler().fit_transform(X)
    classes = np.unique(y)
    cents = [Xs[y == c].mean(axis=0) for c in classes]
    within = np.mean([Xs[y == c].std() for c in classes])
    dists = [np.linalg.norm(cents[i] - cents[j])
             for i in range(len(classes)) for j in range(i + 1, len(classes))]
    return np.mean(dists) / (within + 1e-9)


def compactness(X, y):
    """Mean within-class feature std, standardised X. Lower = tighter classes."""
    Xs = StandardScaler().fit_transform(X)
    return np.mean([Xs[y == c][:, f].std()
                    for c in np.unique(y) for f in range(X.shape[1])])


def modality(X, y, max_components=6):
    """Mean number of Gaussian mixture components per class chosen by BIC."""
    Xs = StandardScaler().fit_transform(X)
    best = []
    for c in np.unique(y):
        Xc = Xs[y == c]
        if len(Xc) < max_components * 2:
            best.append(1)
            continue
        bics = []
        for n in range(1, max_components + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    gm = GaussianMixture(n, random_state=42, n_init=3).fit(Xc)
                bics.append((gm.bic(Xc), n))
            except Exception:
                break
        if bics:
            best.append(min(bics)[1])
    return np.mean(best) if best else np.nan


def profile(X, y):
    """All structural measures for one dataset."""
    return {
        "heterogeneity": scale_heterogeneity(X),
        "skewness": skewness(X, y),
        "kurtosis": excess_kurtosis(X, y),
        "separability": separability(X, y),
        "compactness": compactness(X, y),
        "modality": modality(X, y),
    }
