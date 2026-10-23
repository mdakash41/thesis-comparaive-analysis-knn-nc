# Revisiting KNN and Centroid-Based Classification

Code for the MSc thesis *Revisiting KNN and Centroid-Based Classification: How Dataset Structure Influences Comparative Performance* (ATU Galway).

The thesis revisits Mladenova & Valova (IEEE HORA 2022), which compared KNN with the nearest centroid neighbourhood rule (NCN) on raw, unscaled features. We replicate that comparison, repeat it with features standardised inside each training fold, test why the gap changes, and add the nearest centroid classifier (NC) as a global reference model.

## Project structure

```
knn_nc_research/
├── main.py                 Runs every experiment, then builds the figures
├── requirements.txt
├── data/                   14 UCI datasets as CSV (Breast cancer is loaded from scikit-learn)
├── src/
│   ├── paths.py            data/ and results/ locations
│   ├── datasets.py         Dataset list, loading and preprocessing
│   ├── classifiers.py      NCN (Sanchez, Pla and Ferri, 1997)
│   ├── evaluation.py       Repeated stratified CV, corrected resampled t-test
│   └── measures.py         Scale heterogeneity, skewness, kurtosis, separability,
│                           compactness, modality
├── experiments/
│   ├── exp1_scaling.py                     Thesis 5.1-5.3, Table 5.1, Fig 5.1-5.2
│   ├── exp2_mechanism.py                   Thesis 5.4, Fig 5.3
│   ├── exp3_structural_profile.py          Thesis Table 4.1, 5.5, Fig 5.4
│   ├── exp4_skewness_fixed_separation.py   Thesis 5.6, Table 5.2, Fig 5.5
│   ├── exp5_separability_sweep.py          Thesis 5.7, Table 5.3
│   └── make_figures.py                     All figures, from results/*.csv
├── results/                CSV outputs and results/figures/
└── archive/                Earlier exploratory work, not used for the thesis numbers
```

## How to run

```bash
pip install -r requirements.txt
python main.py                              # everything (Thyroid with exact NCN takes a while)
python -m experiments.exp1_scaling          # or one experiment at a time
python -m experiments.make_figures          # rebuild figures from existing CSVs
```

Run commands from the project root so that `src` can be imported.

## Experiment to thesis map

| Script | What it does | Thesis |
|---|---|---|
| `exp1_scaling` | KNN vs NCN on raw and standardised features, 10x10-fold CV, corrected t-test, Bonferroni, Wilcoxon, heterogeneity vs shrinkage correlation | Table 5.1, Fig 5.1, Fig 5.2 |
| `exp2_mechanism` | Synthetic sweep that scales one weak feature; Wine k sweep | Fig 5.3 |
| `exp3_structural_profile` | Structural measures per dataset; KNN, NCN, NC on standardised features; correlations with the gap; leave-Thyroid-out check | Table 4.1, Fig 5.4 |
| `exp4_skewness_fixed_separation` | Log-normal classes with separability calibrated to a fixed value | Table 5.2, Fig 5.5 |
| `exp5_separability_sweep` | Symmetric Gaussian classes, distance between centres varied | Table 5.3 |

## Method notes

- k = 5 and Euclidean distance throughout, unless stated (k sweep).
- The scaler is always fitted on the training fold only.
- NCN is exact on real data. The synthetic experiments restrict the NCN search to the 60 nearest candidates to keep run time down.
- When the smallest class has fewer than ten members, the number of folds is reduced to that size (minimum two).
- Preprocessing: rows with missing values are dropped, non-numeric attributes are dropped, labels are integer-encoded.

## Data

| File | Target column | Note |
|---|---|---|
| ecoli.csv, glass.csv, hepatitis.csv, ionosphere.csv, iris.csv, lenses.csv, lung_cancer.csv, soybean.csv, wine.csv | `target` | UCI repository |
| absenteeism_at_work.csv | `Absenteeism time in hours` | `;` separated |
| thyroid.csv | `target` | UCI thyroid ANN data, `ann-train` and `ann-test` combined |
| cryotherapy.csv | `Result_of_Treatment` | |
| stars.csv | `Type` | |
| diabetes.csv | `Outcome` | Pima Indians diabetes |
| (scikit-learn) | `target` | Breast Cancer Wisconsin Diagnostic |

SRBCT, Beavers and Algerian Forest Fires from the original study are not included.

## archive/

Kept for the record, not used for any number in the thesis:

- `controlled_dataset_analysis.ipynb` - first KNN vs NC synthetic sweeps (skewness, kurtosis, separability, compactness, modality). The skewness sweep there changes separation together with skewness, which is why it was redesigned as `exp4`. The kurtosis, compactness and modality sweeps are the ones the thesis sets aside in Section 5.7.
- `verify_dataset.py` and `verify_*.png` - visual checks of those early synthetic datasets.
