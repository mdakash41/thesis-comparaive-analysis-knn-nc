# -*- coding: utf-8 -*-
import warnings
import numpy as np
from scipy import stats
from scipy.stats import skew, kurtosis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd

def ncn_predict(X_train, y_train, X_test, k=5, candidate_pool=200):
    predictions = []

    for x_query in X_test:
        # Pre-filter: keep only the closest `candidate_pool` points
        # This makes NCN feasible on large datasets
        if candidate_pool is not None and len(X_train) > candidate_pool:
            dists_all = np.linalg.norm(X_train - x_query, axis=1)
            pool_idx  = np.argsort(dists_all)[:candidate_pool]
            X_pool    = X_train[pool_idx]
            y_pool    = y_train[pool_idx]
        else:
            X_pool = X_train
            y_pool = y_train
            pool_idx = np.arange(len(X_train))

        # Greedy NCN selection
        remaining = list(range(len(X_pool)))
        selected  = []

        for _ in range(min(k, len(X_pool))):
            best_idx  = None
            best_dist = np.inf

            for idx in remaining:
                candidate_set = selected + [idx]
                centroid      = X_pool[candidate_set].mean(axis=0)
                dist          = np.linalg.norm(centroid - x_query)
                if dist < best_dist:
                    best_dist = dist
                    best_idx  = idx

            selected.append(best_idx)
            remaining.remove(best_idx)

        # Majority vote
        neighbour_labels = y_pool[selected]
        pred = stats.mode(neighbour_labels, keepdims=True)[0][0]
        predictions.append(pred)

    return np.array(predictions)


# ── Prepare ───────────────────────────────────────────────────────────────────

def prepare(df, target_col, dataset_name):
    df = df.dropna().reset_index(drop=True)
    feature_df       = df.drop(columns=[target_col])
    numeric_df       = feature_df.select_dtypes(include=[np.number])
    non_numeric_dropped = [c for c in feature_df.columns
                           if c not in numeric_df.columns]
    all_dropped = [target_col] + non_numeric_dropped
    print(f"Dataset: {dataset_name} | Dropped Columns: {', '.join(all_dropped)}")
    X = numeric_df.values
    y = LabelEncoder().fit_transform(df[target_col].values)
    return X, y


def evaluate_cv(X, y, k=5, n_splits=10, n_repeats=10, ncn_pool=200):
    """
    Cross-validated comparison of three classifiers:
        KNN : Traditional k-nearest neighbors (Euclidean distance)
        NCN : Nearest Centroid Neighborhood (centroid-symmetric selection,
              majority vote) — method used in Mladenova & Valova (2022)
        NC  : Nearest Centroid Classifier (assigns to nearest class mean)

    All three use identical folds and StandardScaler.

    Parameters
    ----------
    ncn_pool : int
        Candidate pool size for NCN (speeds up large datasets).
        Set to None for exact NCN (slow on n > 500).
    """
    min_class = np.bincount(y).min()
    n_splits  = min(n_splits, min_class)
    if n_splits <= 1:
        n_splits = 2

    knn_accs = []
    ncn_accs = []
    nc_accs  = []

    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r)
        for tr, te in skf.split(X, y):
            sc  = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])
            ytr = y[tr]
            yte = y[te]

            # ── KNN ──────────────────────────────────────────────────────────
            knn = KNeighborsClassifier(k)
            knn.fit(Xtr, ytr)
            knn_accs.append(accuracy_score(yte, knn.predict(Xte)))

            # ── NCN ──────────────────────────────────────────────────────────
            ncn_pred = ncn_predict(Xtr, ytr, Xte, k=k,
                                   candidate_pool=ncn_pool)
            ncn_accs.append(accuracy_score(yte, ncn_pred))

            # ── NC (Nearest Centroid Classifier) ─────────────────────────────
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nc = NearestCentroid()
                nc.fit(Xtr, ytr)
                nc_accs.append(accuracy_score(yte, nc.predict(Xte)))

    return (round(np.mean(knn_accs), 4),
            round(np.mean(ncn_accs), 4),
            round(np.mean(nc_accs),  4))


# ── Measure functions (unchanged) ─────────────────────────────────────────────

def measure_skewness(X, y):
    vals = []
    for c in np.unique(y):
        Xc = X[y == c]
        for f in range(Xc.shape[1]):
            col = Xc[:, f]
            if col.std() > 0:
                vals.append(abs(skew(col)))
    return round(np.mean(vals), 4) if vals else np.nan


def measure_kurtosis(X, y):
    vals = []
    for c in np.unique(y):
        Xc = X[y == c]
        for f in range(Xc.shape[1]):
            col = Xc[:, f]
            if col.std() > 0:
                vals.append(abs(kurtosis(col)))
    return round(np.mean(vals), 4) if vals else np.nan


def measure_separability(X, y):
    Xs        = StandardScaler().fit_transform(X)
    classes   = np.unique(y)
    centroids = [Xs[y == c].mean(axis=0) for c in classes]
    within_std = np.mean([Xs[y == c].std() for c in classes])
    dists = [
        np.linalg.norm(centroids[i] - centroids[j])
        for i in range(len(classes))
        for j in range(i + 1, len(classes))
    ]
    return round(np.mean(dists) / (within_std + 1e-9), 4)


def measure_compactness(X, y):
    Xs   = StandardScaler().fit_transform(X)
    vals = []
    for c in np.unique(y):
        Xc = Xs[y == c]
        vals.extend(Xc[:, f].std() for f in range(Xc.shape[1]))
    return round(np.mean(vals), 4)


def measure_modality(X, y, max_components=6):
    Xs       = StandardScaler().fit_transform(X)
    best_ns  = []
    for c in np.unique(y):
        Xc = Xs[y == c]
        if len(Xc) < max_components * 2:
            best_ns.append(1)
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
            best_ns.append(min(bics, key=lambda x: x[0])[1])
    return round(np.mean(best_ns), 2) if best_ns else np.nan


# ── Main entry point ──────────────────────────────────────────────────────────

def run_full_profile(datasets, k=5, ncn_pool=200):
    """
    Parameters
    ----------
    datasets : list of (df, target_col, dataset_name)
    k        : neighbours for KNN and NCN (default 5)
    ncn_pool : candidate pool for NCN speed-up (default 200)
    """
    rows = []

    for df, target_col, name in datasets:
        print(f"\n  Processing {name}...", end=" ", flush=True)
        X, y = prepare(df, target_col, name)

        row = {
            "dataset":      name,
            "n_samples":    X.shape[0],
            "n_features":   X.shape[1],
            "n_classes":    len(np.unique(y)),
            "skewness":     measure_skewness(X, y),
            "kurtosis":     measure_kurtosis(X, y),
            "separability": measure_separability(X, y),
            "compactness":  measure_compactness(X, y),
            "modality":     measure_modality(X, y),
        }

        knn_acc, ncn_acc, nc_acc = evaluate_cv(X, y, k=k, ncn_pool=ncn_pool)

        row["knn_acc"]         = knn_acc
        row["ncn_acc"]         = ncn_acc
        row["nc_acc"]          = nc_acc
        row["gap_ncn_knn"]     = round(ncn_acc - knn_acc, 4)
        row["gap_nc_knn"]      = round(nc_acc  - knn_acc, 4)
        rows.append(row)
        print(f"KNN={knn_acc:.4f}  NCN={ncn_acc:.4f}  NC={nc_acc:.4f}")

    result = pd.DataFrame(rows)
    sep    = "=" * 110

    # ── Distributional measures ───────────────────────────────────────────────
    print(f"\n{sep}")
    print("DISTRIBUTIONAL MEASURES  (skewness & kurtosis on raw X; rest on scaled X)")
    print(sep)
    profile_cols = ["dataset", "n_samples", "n_features", "n_classes",
                    "skewness", "kurtosis", "separability", "compactness", "modality"]
    print(result[profile_cols].to_string(index=False))

    # ── Classifier results ────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("CLASSIFIER RESULTS  (10-fold CV × 10 repeats, k=5)")
    print(f"{'Dataset':<16} {'KNN':>8} {'NCN':>8} {'NC':>8} "
          f"{'NCN-KNN':>10} {'NC-KNN':>10}")
    print("-" * 65)
    for _, row in result.iterrows():
        print(f"{row['dataset']:<16} "
              f"{row['knn_acc']:>8.4f} "
              f"{row['ncn_acc']:>8.4f} "
              f"{row['nc_acc']:>8.4f} "
              f"{row['gap_ncn_knn']:>+10.4f} "
              f"{row['gap_nc_knn']:>+10.4f}")

    # Summary statistics
    print(f"\n  NCN > KNN : {(result['gap_ncn_knn'] >  0.005).sum()}/{len(result)}")
    print(f"  NCN < KNN : {(result['gap_ncn_knn'] < -0.005).sum()}/{len(result)}")
    print(f"  NC  > KNN : {(result['gap_nc_knn']  >  0.005).sum()}/{len(result)}")
    print(f"  NC  < KNN : {(result['gap_nc_knn']  < -0.005).sum()}/{len(result)}")
    print(f"  Mean gap NCN-KNN : {result['gap_ncn_knn'].mean():+.4f}")
    print(f"  Mean gap NC-KNN  : {result['gap_nc_knn'].mean():+.4f}")

    # ── Correlation ───────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print("CORRELATION — each measure vs gap")
    print(f"{'Measure':<15} {'r(NCN-KNN)':>12} {'r(NC-KNN)':>12}")
    print("-" * 42)
    measures = ["skewness", "kurtosis", "separability", "compactness", "modality"]
    for m in measures:
        col = result[m].dropna()
        if len(col) < 2 or col.std() == 0:
            print(f"  {m:<15}  N/A")
            continue
        r_ncn = col.corr(result.loc[col.index, "gap_ncn_knn"])
        r_nc  = col.corr(result.loc[col.index, "gap_nc_knn"])
        print(f"  {m:<15}  {r_ncn:>+10.3f}   {r_nc:>+10.3f}")

    return result


# ── Run ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    lenses_df      = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\lenses.csv")
    lung_cancer_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\lung_cancer.csv")
    soybean_df     = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\soybean.csv")
    cryotherapy    = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\cryotherapy.csv")
    irish_df       = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\iris.csv")
    hepatitis_df   = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\hepatitis.csv")
    wine_df        = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\wine.csv")
    glass_df       = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\glass.csv")
    thyroid_df     = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\thyroid.csv")
    stars          = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\stars.csv")
    ecoli_df       = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\ecoli.csv")
    ionosphere_df  = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\ionosphere.csv")
    absence_df     = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\absenteeism_at_work.csv", sep=';')
    pima_diabetes  = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\diabetes.csv")

    run_full_profile([
        (ecoli_df,      "target",                  "Ecoli"),
        (glass_df,      "target",                  "Glass"),
        (hepatitis_df,  "target",                  "Hepatitis"),
        (ionosphere_df, "target",                  "Ionosphere"),
        (irish_df,      "target",                  "Iris"),
        (lenses_df,     "target",                  "Lenses"),
        (lung_cancer_df,"target",                  "Lungcancer"),
        (soybean_df,    "target",                  "Soybean"),
        (wine_df,       "target",                  "Wine"),
        (absence_df,    "Absenteeism time in hours","Absence"),
        (thyroid_df,    "target",                  "Thyroid"),
        (cryotherapy,   "Result_of_Treatment",     "Cryotherapy"),
        (stars,         "Type",                    "Stars"),
        (pima_diabetes, "Outcome",                 "Diabetes"),
    ], k=5, ncn_pool=200)
