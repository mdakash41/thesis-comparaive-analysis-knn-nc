"""Nearest centroid neighbourhood (NCN) rule, Sanchez, Pla and Ferri (1997)."""

import numpy as np


def ncn_predict(X_train, y_train, X_test, k=5, pool=None):
    """
    For each query, pick k neighbours one at a time so that the centroid of
    the chosen set stays as close as possible to the query, then majority vote.

    pool: if set, only the `pool` nearest training points are candidates
          (an approximation used in the synthetic runs). None = exact NCN.
    With k=1 the prediction is identical to 1-NN.
    """
    preds = np.empty(len(X_test), dtype=y_train.dtype)
    for i, xq in enumerate(X_test):
        if pool is not None and pool < len(X_train):
            dist = np.linalg.norm(X_train - xq, axis=1)
            idx = np.argsort(dist, kind="stable")[:pool]
            P, yP = X_train[idx], y_train[idx]
        else:
            P, yP = X_train, y_train

        available = np.ones(len(P), bool)
        running_sum = np.zeros(P.shape[1])
        chosen = []
        for step in range(min(k, len(P))):
            centroids = (running_sum + P) / (step + 1)
            dist = np.linalg.norm(centroids - xq, axis=1)
            dist[~available] = np.inf
            j = int(np.argmin(dist))
            chosen.append(j)
            available[j] = False
            running_sum += P[j]

        labels, counts = np.unique(yP[chosen], return_counts=True)
        preds[i] = labels[np.argmax(counts)]
    return preds
