import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, skew
from sklearn.datasets import make_blobs
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


# ═════════════════════════════════════════════════════════════════════════════
# NCN IMPLEMENTATION
# Reference: Mladenova & Valova (2022) — "Nearest Centroid Neighborhood"
#
# Core idea: instead of picking the k nearest neighbors (KNN),
# pick k/2 PAIRS where each pair (n_i, n_j) is symmetric around x:
#   n_j ≈ 2x - n_i   (mirror of n_i through the test point x)
# This makes the neighborhood geometrically balanced.
#
# Algorithm for each test point x:
#   1. For every training point i, find its ideal mirror: 2x - X[i]
#   2. Find j = closest training point to that mirror (≠ i)
#   3. Score the pair by dist(x, X[i])  — same ranking as KNN
#   4. Greedily pick top-scoring non-overlapping pairs until k points chosen
#   5. Majority vote on the k chosen points
# ═════════════════════════════════════════════════════════════════════════════

class NearestCentroidNeighborhood:
    """
    Nearest Centroid Neighborhood (NCN) classifier.
    Implements the symmetric neighborhood from Mladenova & Valova (2022).
    n_neighbors is forced even so pairs are always balanced.
    """

    def __init__(self, n_neighbors=4):
        self.k = n_neighbors + (n_neighbors % 2)   # force even

    def fit(self, X, y):
        self.X_ = X.astype(float)
        self.y_ = y.copy()
        return self

    def predict(self, X):
        return np.array([self._pred_one(x) for x in X])

    def _pred_one(self, x):
        d_to_x = np.sum((self.X_ - x) ** 2, axis=1)   # squared L2 to x

        # build candidate pairs: (score, anchor_idx, partner_idx)
        pairs = []
        for i in range(len(self.X_)):
            mirror   = 2 * x - self.X_[i]
            d_mirror = np.sum((self.X_ - mirror) ** 2, axis=1)
            d_mirror[i] = np.inf                        # exclude self
            j = int(np.argmin(d_mirror))
            pairs.append((d_to_x[i], i, j))

        pairs.sort(key=lambda t: t[0])                  # rank by closeness of anchor

        chosen, used = [], set()
        for _, i, j in pairs:
            if len(chosen) >= self.k:
                break
            if i in used or j in used:
                continue
            chosen += [i, j]
            used   |= {i, j}

        if not chosen:
            return self.y_[int(np.argmin(d_to_x))]     # fallback: 1-NN

        votes = self.y_[chosen]
        return int(np.argmax(np.bincount(votes, minlength=int(self.y_.max()) + 1)))


# ═════════════════════════════════════════════════════════════════════════════
# SHARED HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def evaluate_with_cv(X, y, k=4, n_splits=10, n_repeats=5):
    """5 × 10-fold CV — enough for reliable estimates while keeping NCN fast."""
    min_class = np.bincount(y).min()
    n_splits  = min(n_splits, min_class)

    knn_accs, ncn_accs = [], []
    for repeat in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=repeat)
        for tr, te in skf.split(X, y):
            scaler = StandardScaler()
            X_tr   = scaler.fit_transform(X[tr])
            X_te   = scaler.transform(X[te])

            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X_tr, y[tr])
            knn_accs.append(accuracy_score(y[te], knn.predict(X_te)))

            ncn = NearestCentroidNeighborhood(n_neighbors=k)
            ncn.fit(X_tr, y[tr])
            ncn_accs.append(accuracy_score(y[te], ncn.predict(X_te)))

    return np.mean(knn_accs), np.mean(ncn_accs)


def measure_avg_skewness(X, y):
    vals = []
    for c in np.unique(y):
        Xc = X[y == c]
        for f in range(Xc.shape[1]):
            if Xc[:, f].std() > 0:
                vals.append(abs(skew(Xc[:, f])))
    return round(np.mean(vals), 4) if vals else float('nan')


def plot_results(df, x_col, x_label, title):
    plt.figure(figsize=(11, 6))
    plt.plot(df[x_col], df['knn_acc'], marker='o', linewidth=2,
             label='KNN (k=4)', color='royalblue')
    plt.plot(df[x_col], df['ncn_acc'], marker='s', linewidth=2,
             label='NCN  (k=4)', color='darkorange')
    plt.xlabel(x_label, fontsize=12)
    plt.ylabel('Accuracy (10-fold CV × 5 repeats)', fontsize=12)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.ylim(0.4, 1.05)
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1 — SKEWNESS
# NCN assumes symmetric pairs around x are meaningful.
# When the distribution is skewed, points on one side of x are much
# denser than the other → the mirror partner is in a low-density region
# → the pair is unrepresentative → accuracy degrades.
# ═════════════════════════════════════════════════════════════════════════════

def generate_skewed(n_samples, sigma, seed=42):
    sep = 1.5 * sigma
    X0  = lognorm.rvs(s=sigma, size=(n_samples // 2, 4), random_state=seed)
    X1  = lognorm.rvs(s=sigma, size=(n_samples // 2, 4), random_state=seed + 1)
    X1 += sep
    X   = np.vstack([X0, X1])
    y   = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    return X, y

def run_skewness_experiment(n_samples=200):
    sigma_values = [0.1, 0.3, 0.5, 0.8, 1.2, 1.5, 1.8, 2.1, 2.4, 2.8, 3.0]
    results = []
    for sigma in sigma_values:
        X, y             = generate_skewed(n_samples, sigma)
        avg_skew         = measure_avg_skewness(X, y)
        knn_acc, ncn_acc = evaluate_with_cv(X, y)
        results.append({
            'sigma':            sigma,
            'avg_abs_skewness': avg_skew,
            'knn_acc':          round(knn_acc, 4),
            'ncn_acc':          round(ncn_acc, 4),
            'gap (NCN-KNN)':    round(ncn_acc - knn_acc, 4),
        })
    df_res = pd.DataFrame(results)
    print("=== SKEWNESS EXPERIMENT (KNN vs NCN) ===")
    print(df_res.to_string(index=False))
    plot_results(df_res, 'avg_abs_skewness',
                 'Mean Absolute Skewness (per class, per feature)',
                 'KNN vs NCN — Skewness Experiment')
    return df_res


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2 — SEPARABILITY
# NCN needs to find a partner from the opposite class near the mirror point.
# Low separability → opposite-class points are far from where the mirror
# lands → NCN chooses a geometrically wrong partner → worse than KNN.
# High separability → both trivially succeed.
# ═════════════════════════════════════════════════════════════════════════════

def run_separability_experiment(n_samples=200):
    distance_values = [0.3, 0.5, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 4.0]
    results = []
    for dist in distance_values:
        centres          = [[0, 0, 0, 0], [dist, 0, 0, 0]]
        X, y             = make_blobs(n_samples=n_samples, n_features=4,
                                      centers=centres, cluster_std=1.0, random_state=42)
        knn_acc, ncn_acc = evaluate_with_cv(X, y)
        results.append({
            'centre_distance': dist,
            'knn_acc':         round(knn_acc, 4),
            'ncn_acc':         round(ncn_acc, 4),
            'gap (NCN-KNN)':   round(ncn_acc - knn_acc, 4),
        })
    df_res = pd.DataFrame(results)
    print("\n=== SEPARABILITY EXPERIMENT (KNN vs NCN) ===")
    print(df_res.to_string(index=False))
    plot_results(df_res, 'centre_distance',
                 'Distance Between Class Centres',
                 'KNN vs NCN — Separability Experiment')
    return df_res


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3 — MODALITY
# NCN's symmetry assumption breaks when a class has multiple modes.
# The mirror of a neighbor in mode-A may land near mode-B of the same class,
# so the best partner comes from the wrong region → noisy neighborhood.
# class_sep is kept small (2.0) so the task stays genuinely hard.
# ═════════════════════════════════════════════════════════════════════════════

def generate_multimodal(n_samples, n_modes, class_sep=2.0, mode_spread=3.5, seed=42):
    rng     = np.random.default_rng(seed)
    X_parts, y_parts = [], []
    for cls in range(2):
        n          = n_samples // 2
        n_per_mode = max(1, n // n_modes)
        centres    = [
            [cls * class_sep, (m - n_modes / 2) * mode_spread, 0, 0]
            for m in range(n_modes)
        ]
        chunks = [rng.normal(loc=c, scale=0.7, size=(n_per_mode, 4))
                  for c in centres]
        X_parts.append(np.vstack(chunks))
        y_parts.append(np.full(n_per_mode * n_modes, cls))
    return np.vstack(X_parts), np.concatenate(y_parts)

def run_modality_experiment(n_samples=200):
    mode_values = [1, 2, 3, 4, 5, 6, 8, 10]
    results = []
    for n_modes in mode_values:
        X, y             = generate_multimodal(n_samples, n_modes)
        knn_acc, ncn_acc = evaluate_with_cv(X, y)
        results.append({
            'n_modes':       n_modes,
            'knn_acc':       round(knn_acc, 4),
            'ncn_acc':       round(ncn_acc, 4),
            'gap (NCN-KNN)': round(ncn_acc - knn_acc, 4),
        })
    df_res = pd.DataFrame(results)
    print("\n=== MODALITY EXPERIMENT (KNN vs NCN) ===")
    print(df_res.to_string(index=False))
    plot_results(df_res, 'n_modes',
                 'Number of Modes per Class',
                 'KNN vs NCN — Modality Experiment')
    return df_res


# ─── Run all ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_skewness_experiment()
    run_separability_experiment()
    run_modality_experiment()