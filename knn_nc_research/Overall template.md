# Comparative Dataset Analysis — NC Suitability Assessment

This document provides a consolidated comparison of all nine datasets analysed for Nearest Centroid (NC) suitability. Each dataset has been evaluated across five criteria: multivariate normality, unimodality, feature skewness, kurtosis, and class separability.

---

## Individual Dataset Analyses

1. [Iris](Result%20analysis-iris.md)
2. [Ecoli](Result%20analysis-ecoli.md)
3. [Glass](Result%20analysis-glass.md)
4. [Hepatitis](Result%20analysis-hepatitis.md)
5. [Ionosphere](Result%20analysis-ionosphere.md)
6. [Lenses](Result%20analysis-lenses.md)
7. [Lung Cancer](Result%20analysis-lung-cancer.md)
8. [Soybean](Result%20analysis-soybean.md)
9. [Wine](Result%20analysis-wine.md)

---

## Dataset Bias Score Methodology

> Each criterion is assigned a score of 1 if the dataset satisfies the NC-favourable condition, and 0 otherwise. The thresholds for each criterion are defined in the individual dataset analyses and summarised below.

### Scoring Criteria

| Criterion | Threshold for NC-Favourable (Score = 1) |
|-----------|----------------------------------------|
| Multivariate Normality | ≥50% of classes satisfy multivariate normality (Henze-Zirkler, α = 0.05) |
| Unimodality | ≥50% of classes satisfy unimodality threshold (≥70% unimodal features per class) |
| Low Skewness | ≥50% of classes satisfy skewness threshold (≥60% near-normal features per class) |
| Near-Normal Kurtosis | ≥50% of classes satisfy kurtosis threshold (≥60% near-normal features per class) |
| High Separability | Average separability ratio ≥ 1.5 AND ≥50% of class pairs achieve high separability |

### Score Interpretation

| Total Score | Interpretation |
|-------------|----------------|
| 5 | Strongly NC-favourable |
| 4 | NC-favourable |
| 3 | Neutral |
| 2 | Weakly NC-favourable |
| 0–1 | Not NC-favourable |

---

## Summary Comparison Table

| Dataset | Normality | Unimodality | Skewness | Kurtosis | Separability | Score | NC-Favourable? |
|---------|-----------|-------------|----------|----------|--------------|-------|----------------|
| Iris | Partial (2/3) | Mostly (3/3) | High (3/3) | High (3/3) | High (4.62) | **5/5** | ✓ Strongly NC-favourable |
| Wine | Mostly (2/3) | Strong (3/3) | High (3/3) | High (3/3) | Exceptional (13.29) | **5/5** | ✓ Strongly NC-favourable |
| Ecoli | No (0/6) | Mostly (6/8) | Partial (3/8) | Partial (3/8) | High (3.19) | **3/5** | ~ Neutral |
| Glass | Partial (2/6) | Strong (6/6) | Mostly no (2/6) | Partial (3/6) | Mixed (2.23) | **2/5** | ✗ Weakly NC-favourable |
| Hepatitis | No (0/2) | No (0/2) | No (0/2) | Yes (2/2) | High (2.47) | **2/5** | ✗ Weakly NC-favourable |
| Ionosphere | No (0/2) | Mixed (1/2) | Yes (2/2) | Strong (2/2) | Moderate (1.36) | **2/5** | ✗ Weakly NC-favourable |
| Lenses | No (0/2) | No (0/3) | Mostly no (1/3) | Partial (1/3) | Mixed (1.99) | **0/5** | ✗ Not NC-favourable |
| Lung Cancer | No (0/3) | No (0/3) | Mostly no (1/3) | Partial (2/3) | High (3.08) | **1/5** | ✗ Not NC-favourable |
| Soybean | No (0/4) | Yes (4/4) | No (0/4) | No (0/4) | Exceptional (14.28) | **2/5** | ✗ Weakly NC-favourable |

---

## Detailed Scoring Breakdown

### Iris — Score: 5/5 (Strongly NC-favourable)

| Criterion | Classes Meeting Threshold | Score |
|-----------|---------------------------|-------|
| Normality | 2/3 (66.7%) | 1 |
| Unimodality | 3/3 (100%) | 1 |
| Skewness | 3/3 (100%) | 1 |
| Kurtosis | 3/3 (100%) | 1 |
| Separability | Avg 4.62, 3/3 pairs HIGH | 1 |
| **Total** | | **5/5** |

### Wine — Score: 5/5 (Strongly NC-favourable)

| Criterion | Classes Meeting Threshold | Score |
|-----------|---------------------------|-------|
| Normality | 2/3 (66.7%) | 1 |
| Unimodality | 3/3 (100%) | 1 |
| Skewness | 3/3 (100%) | 1 |
| Kurtosis | 3/3 (100%) | 1 |
| Separability | Avg 13.29, 3/3 pairs HIGH | 1 |
| **Total** | | **5/5** |

### Ecoli — Score: 3/5 (Neutral)

| Criterion | Classes Meeting Threshold | Score |
|-----------|---------------------------|-------|
| Normality | 0/6 (0%) — 2 classes N/A | 0 |
| Unimodality | 6/8 (75%) | 1 |
| Skewness | 3/8 (37.5%) | 0 |
| Kurtosis | 3/8 (37.5%) | 0 |
| Separability | Avg 3.19, 26/28 pairs HIGH | 1 |
| **Total** | | **3/5** |

### Glass — Score: 2/5 (Weakly NC-favourable)

| Criterion | Classes Meeting Threshold | Score |
|-----------|---------------------------|-------|
| Normality | 2/6 (33.3%) | 0 |
| Unimodality | 6/6 (100%) | 1 |
| Skewness | 2/6 (33.3%) | 0 |
| Kurtosis | 3/6 (50%) | 1 |
| Separability | Avg 2.23, 10/15 pairs HIGH | 0 |
| **Total** | | **2/5** |

### Hepatitis — Score: 2/5 (Weakly NC-favourable)

| Criterion | Classes Meeting Threshold | Score |
|-----------|---------------------------|-------|
| Normality | 0/2 (0%) | 0 |
| Unimodality | 0/2 (0%) | 0 |
| Skewness | 0/2 (0%) | 0 |
| Kurtosis | 2/2 (100%) | 1 |
| Separability | Avg 2.47, 1/1 pairs HIGH | 1 |
| **Total** | | **2/5** |

### Ionosphere — Score: 2/5 (Weakly NC-favourable)

| Criterion | Classes Meeting Threshold | Score |
|-----------|---------------------------|-------|
| Normality | 0/2 (0%) | 0 |
| Unimodality | 1/2 (50%) | 1 |
| Skewness | 2/2 (100%) | 1 |
| Kurtosis | 2/2 (100%) | 1 |
| Separability | Avg 1.36, 0/1 pairs HIGH | 0 |
| **Total** | | **2/5** |

### Lenses — Score: 0/5 (Not NC-favourable)

| Criterion | Classes Meeting Threshold | Score |
|-----------|---------------------------|-------|
| Normality | 0/2 (0%) — 1 class N/A | 0 |
| Unimodality | 0/3 (0%) | 0 |
| Skewness | 1/3 (33.3%) | 0 |
| Kurtosis | 1/3 (33.3%) | 0 |
| Separability | Avg 1.99, 1/3 pairs HIGH | 0 |
| **Total** | | **0/5** |

### Lung Cancer — Score: 1/5 (Not NC-favourable)

| Criterion | Classes Meeting Threshold | Score |
|-----------|---------------------------|-------|
| Normality | 0/3 (0%) | 0 |
| Unimodality | 0/3 (0%) | 0 |
| Skewness | 1/3 (33.3%) | 0 |
| Kurtosis | 2/3 (66.7%) | 1 |
| Separability | Avg 3.08, 3/3 pairs HIGH | 1 |
| **Total** | | **1/5** |

### Soybean — Score: 2/5 (Weakly NC-favourable)

| Criterion | Classes Meeting Threshold | Score |
|-----------|---------------------------|-------|
| Normality | 0/4 (0%) | 0 |
| Unimodality | 4/4 (100%) | 1 |
| Skewness | 0/4 (0%) | 0 |
| Kurtosis | 0/4 (0%) | 0 |
| Separability | Avg 14.28, 6/6 pairs HIGH | 1 |
| **Total** | | **2/5** |

---

## Distribution of Scores

| Score | Interpretation | Datasets |
|-------|----------------|----------|
| 5/5 | Strongly NC-favourable | Iris, Wine |
| 4/5 | NC-favourable | — |
| 3/5 | Neutral | Ecoli |
| 2/5 | Weakly NC-favourable | Glass, Hepatitis, Ionosphere, Soybean |
| 1/5 | Not NC-favourable | Lung Cancer |
| 0/5 | Not NC-favourable | Lenses |

---

## Key Observations

1. **Only two datasets (Iris, Wine) achieve the highest NC-favourable rating.** These datasets satisfy all five criteria and represent benchmark cases where NC assumptions are comprehensively met.

2. **No dataset scored 4/5**, suggesting that when NC assumptions begin to fail, they tend to fail across multiple criteria simultaneously.

3. **Separability is the most commonly satisfied criterion** (7 of 9 datasets achieve high separability), while **multivariate normality is the least satisfied** (only 2 of 9 datasets meet the threshold).

4. **The Soybean dataset illustrates a critical insight:** exceptional separability (14.28) coexists with severe violations of skewness and kurtosis, demonstrating that class separation alone does not guarantee NC suitability.

5. **Small sample sizes and categorical features** (Lenses, Lung Cancer, Hepatitis) consistently produce the least NC-favourable profiles, highlighting fundamental limitations of centroid-based methods on such data.

---

## Conclusion

This comparative analysis demonstrates that datasets commonly used to benchmark KNN and NC classifiers exhibit vastly different structural properties with respect to NC assumptions. The fact that only two of nine datasets are strongly NC-favourable reinforces the central thesis: **observing similar performance between KNN and NC on a given dataset should not be interpreted as evidence of general algorithmic equivalence**. Rather, performance parity is frequently a consequence of dataset-specific structural characteristics that either favour or disadvantage centroid-based classification.