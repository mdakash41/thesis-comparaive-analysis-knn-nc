# -*- coding: utf-8 -*-
import warnings
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pandas as pd


def prepare(df, target_col, dataset_name):
    df = df.dropna().reset_index(drop=True)

    # 2. Isolate the feature dataframe by dropping the target column
    feature_df = df.drop(columns=[target_col])

    # 3. Identify columns dropped because they are not numeric
    numeric_df = feature_df.select_dtypes(include=[np.number])
    non_numeric_dropped = [col for col in feature_df.columns if col not in numeric_df.columns]

    # 4. Combine all dropped columns for logging (Target + Non-numeric)
    all_dropped_cols = [target_col] + non_numeric_dropped
    dropped_string = ", ".join(all_dropped_cols)

    # 5. Print the formatted result string
    print(f"Dataset: {dataset_name} | Dropped Columns: {dropped_string}")

    # 6. Extract raw values for your ML models
    X = numeric_df.values
    y = LabelEncoder().fit_transform(df[target_col].values)
    return X, y



def evaluate_cv(X, y, k=5, n_splits=10, n_repeats=10):
    min_class = np.bincount(y).min()
    n_splits = min(n_splits, min_class)
    if n_splits <= 1:
        n_splits = 2
    knn_accs, nc_accs = [], []
    for r in range(n_repeats):
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=r)
        for tr, te in skf.split(X, y):
            sc = StandardScaler()
            Xtr = sc.fit_transform(X[tr])
            Xte = sc.transform(X[te])

            knn_accs.append(
                accuracy_score(y[te],
                    KNeighborsClassifier(k).fit(Xtr, y[tr]).predict(Xte))
            )
            # suppress the zero-std warning — it is a known data property
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                nc_accs.append(
                    accuracy_score(y[te],
                        NearestCentroid().fit(Xtr, y[tr]).predict(Xte))
                )
    return np.mean(knn_accs), np.mean(nc_accs)


# ── measure functions (all on RAW unscaled X) ─────────────────────────────────

def measure_skewness(X, y):
    """
    Mean absolute skewness across all features and classes.
    Computed on raw X — scaling removes the shape we want to measure.
    Skips constant features (std=0) to avoid NaN.
    """
    vals = []
    for c in np.unique(y):
        Xc = X[y == c]
        for f in range(Xc.shape[1]):
            col = Xc[:, f]
            if col.std() > 0:                  # skip constant columns
                vals.append(abs(skew(col)))
    return round(np.mean(vals), 4) if vals else np.nan


def measure_kurtosis(X, y):
    """
    Mean absolute excess kurtosis across all features and classes.
    Computed on raw X. Skips constant features.
    """
    vals = []
    for c in np.unique(y):
        Xc = X[y == c]
        for f in range(Xc.shape[1]):
            col = Xc[:, f]
            if col.std() > 0:
                vals.append(abs(kurtosis(col)))
    return round(np.mean(vals), 4) if vals else np.nan


def measure_separability(X, y):
    """
    Mean pairwise centroid distance normalised by mean within-class std.
    Computed on StandardScaled X so feature scale does not bias distances.
    """
    Xs = StandardScaler().fit_transform(X)
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
    """
    Mean within-class std across all features and classes.
    Computed on StandardScaled X. Lower = more compact.
    """
    Xs = StandardScaler().fit_transform(X)
    vals = []
    for c in np.unique(y):
        Xc = Xs[y == c]
        vals.extend(Xc[:, f].std() for f in range(Xc.shape[1]))
    return round(np.mean(vals), 4)


def measure_modality(X, y, max_components=6):
    """
    Per class: fit GMMs 1..max_components, pick best by BIC.
    Returns mean best n_components across classes.
    Higher = more multimodal. Computed on StandardScaled X.
    """
    Xs = StandardScaler().fit_transform(X)
    best_ns = []
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


# ── main entry point ──────────────────────────────────────────────────────────

def run_full_profile(datasets, k=5):
    """
    Parameters
    ----------
    datasets : list of (df, target_col, dataset_name)
    k        : KNN neighbours (default 5, same as paper)

    Returns
    -------
    pd.DataFrame — one row per dataset, all measures + accuracy
    """
    rows = []
    for df, target_col, name in datasets:
        print(f"  Processing {name}...", end=" ", flush=True)
        X, y = prepare(df, target_col, name)

        row = {
            "dataset":      name,
            "n_samples":    X.shape[0],
            "n_features":   X.shape[1],
            "n_classes":    len(np.unique(y)),
            # --- measures on RAW X ---
            "skewness":     measure_skewness(X, y),
            "kurtosis":     measure_kurtosis(X, y),
            # --- measures on SCALED X ---
            "separability": measure_separability(X, y),
            "compactness":  measure_compactness(X, y),
            "modality":     measure_modality(X, y),
        }

        knn_acc, nc_acc = evaluate_cv(X, y, k=k)
        row["knn_acc"]      = round(knn_acc, 4)
        row["nc_acc"]       = round(nc_acc,  4)
        row["gap (NC-KNN)"] = round(nc_acc - knn_acc, 4)
        rows.append(row)
        print("done")

    result = pd.DataFrame(rows)


    sep = "=" * 100

    print(f"\n{sep}")
    print("DISTRIBUTIONAL MEASURES  (skewness & kurtosis on raw X; rest on scaled X)")
    print(sep)
    profile_cols = ["dataset", "n_samples", "n_features", "n_classes",
                    "skewness", "kurtosis", "separability", "compactness", "modality"]
    print(result[profile_cols].to_string(index=False))

    print(f"\n{sep}")
    print("CLASSIFIER RESULTS  (10-fold CV × 10 repeats, k=5)")
    print(sep)
    print(result[["dataset", "knn_acc", "nc_acc", "gap (NC-KNN)"]].to_string(index=False))

    print(f"\n{sep}")
    print("CORRELATION — each measure vs gap (NC-KNN)")
    print(sep)
    measures = ["skewness", "kurtosis", "separability", "compactness", "modality"]
    for m in measures:
        col = result[m].dropna()
        gap_col = result.loc[col.index, "gap (NC-KNN)"]
        if len(col) >= 2 and col.std() > 0:
            r = col.corr(gap_col)
            direction = "NC worse as ↑" if r < 0 else "NC better as ↑"
            print(f"  {m:<15}  r = {r:+.3f}   ({direction})")
        else:
            print(f"  {m:<15}  r = N/A  (not enough variance)")

    return result

if __name__ == "__main__":

    lenses_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\lenses.csv")
    lung_cancer_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\lung_cancer.csv")
    soybean_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\soybean.csv")
    srbct = None
    cryotherapy = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\cryotherapy.csv")
    beavers = None
    irish_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\iris.csv")
    hepatitis_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\hepatitis.csv")
    wine_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\wine.csv")
    glass_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\glass.csv")
    thyroid_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\thyroid.csv")
    stars = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\stars.csv")
    algerian_fires = None
    ecoli_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\ecoli.csv")
    ionosphere_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\ionosphere.csv")
    breast_cancer = None
    absence_df = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\absenteeism_at_work.csv", sep = ';')
    pima_diabetes = pd.read_csv(r"E:\ATU-Masters\Semester 2 ( research paper )\thesis-comparaive-analysis-knn-nc\knn_nc_research\data\diabetes.csv")

    run_full_profile([
        (ecoli_df, "target", "Ecoli"),
        (glass_df, "target", "Glass"),
        (hepatitis_df, "target", "Hepatitis"),
        (ionosphere_df, "target", "Ionosphere"),
        (irish_df, "target", "Iris"),
        (lenses_df, "target", "Lenses"),
        (lung_cancer_df, "target", "Lungcancer"),
        (soybean_df, "target", "Soybean"),
        (wine_df, "target", "Wine"),
        (absence_df, "Absenteeism time in hours", "Absence"),
        (thyroid_df, "target", "Thyroid"),
        (cryotherapy, "Result_of_Treatment", "Cryotherapy"),
        (stars, "Type", "Stars"),
        (pima_diabetes, "Outcome", "Diabetes"),
    ])

