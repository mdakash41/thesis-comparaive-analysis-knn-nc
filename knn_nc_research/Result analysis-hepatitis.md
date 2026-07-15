# Dataset Analysis Hepatitis

1. Dataset summary
2. Statistical analysis
3. NC suitability assessment
4. Overall bias score (optional)
5. Conclusion

---

## Dataset: Hepatitis
> Definition 1 (NC-favourable class). In this study, a class is termed NC-favourable if it satisfies the threshold defined for the corresponding statistical property (e.g., ≥70% unimodal features, ≥60% low-skewness features, etc.). The term does not imply that NC will necessarily outperform KNN; rather, it indicates that the class exhibits characteristics consistent with the assumptions of the Nearest Centroid classifier.

### 1. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | Hepatitis |
| Samples | 155 |
| Features | 19 |
| Classes | 2 |
| Samples per class | 1: 123, 2: 32 |

### 2. Statistical Analysis

#### 2.1 Multivariate Normality (Henze-Zirkler)

| Class | HZ Statistic | p-value | Normal Distribution |
|-------|--------------|---------|---------------------|
| 1 | 52.0 | 0.0 | ✗ No |
| 2 | 1.1343 | 0.0 | ✗ No |

**Interpretation**

Neither class satisfies multivariate normality. The dataset does not conform to the centroid assumption used by Nearest Centroid. The majority class (1) shows extreme deviation, likely influenced by the mixed data types (categorical and continuous) present in this medical dataset.

#### 2.2 Unimodality (Dip Test)
> Note. Nearest Centroid (NC) assumes that each class can be reasonably represented by a single centroid. To evaluate this assumption, the percentage of unimodal features within each class is calculated using Hartigan's Dip Test (α = 0.05). A class is considered NC-favourable when 70% or more of its features are unimodal; otherwise, it is considered to exhibit multimodal structure that may reduce centroid representativeness.

| Class | Unimodal Features | Interpretation |
|-------|-------------------|----------------|
| 1 | 63.2% | Only 63% of features are unimodal — multimodal structure detected. |
| 2 | 26.3% | Only 26% of features are unimodal — multimodal structure detected. |

**Interpretation**

Neither class meets the 70% unimodality threshold. Class 2 is particularly affected, with only 26.3% of features exhibiting unimodal distributions. This indicates substantial multimodal structure, likely attributable to the binary and categorical features present in the dataset, which undermines the single-centroid assumption of NC.

#### 2.3 Feature Skewness
> Note. NC represents each class using its arithmetic mean (centroid). Highly skewed feature distributions can shift the mean away from the region containing most observations, making the centroid less representative. For each class, the percentage of features whose absolute skewness is below 0.5 is calculated. A class is considered NC-favourable when 60% or more of its features have low skewness; otherwise, asymmetric distributions are considered likely to reduce centroid reliability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| 1 | 52.6% | Only 53% of features have low skewness — asymmetric distributions detected. |
| 2 | 52.6% | Only 53% of features have low skewness — asymmetric distributions detected. |

**Interpretation**

Both classes fall below the 60% NC-favourable threshold for low skewness. With only 52.6% near-normal features in each class, asymmetric distributions are prevalent. This suggests that centroids may be shifted away from the central tendency of the data, potentially reducing classification reliability.

#### 2.4 Kurtosis
> Note. NC performs best when feature distributions are not dominated by heavy tails or extreme outliers, as such observations can distort the centroid. For each class, the percentage of features whose kurtosis falls within the predefined near-normal range is calculated. A class is considered NC-favourable when 60% or more of its features exhibit near-normal kurtosis; otherwise, heavy-tailed distributions are considered likely to reduce centroid stability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| 1 | 73.7% | 74% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| 2 | 78.9% | 79% of features have near-normal kurtosis — tail behaviour is NC-favourable. |

**Interpretation**

Both classes satisfy the NC-favourable threshold for near-normal kurtosis, indicating that tail behaviour is generally well-behaved. Extreme observations are unlikely to exert disproportionate influence on centroid estimation, which partially mitigates concerns raised by other criteria.

#### 2.5 Class Separability
> Note. NC assigns a sample to the nearest class centroid. Therefore, its performance depends on the distance between class centroids relative to the spread of each class. The separability ratio is defined as the inter-class centroid distance divided by the average within-class dispersion. Larger ratios indicate better-separated classes and are therefore more favourable for NC. The following interpretation is used throughout this study:

- Ratio < 1.0: Low separability (NC-unfavourable)
- 1.0 ≤ Ratio < 1.5: Moderate separability
- Ratio ≥ 1.5: High separability (NC-favourable)

| Class Pair | Ratio | Separation |
|------------|-------|-------------|
| 1 vs 2 | 2.473 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |

**Average Separability Ratio**

2.473

**Interpretation**

The single class pair exhibits high separability, with a ratio of 2.473 well above the NC-favourable threshold. This indicates that despite the internal structural issues within each class, the two classes are sufficiently distinct in the feature space for a centroid-based classifier to distinguish them.

### 3. Overall Dataset Assessment

| Criterion | Result | Supports NC? |
|-----------|--------|--------------|
| Multivariate normality | Not satisfied | ✗ |
| Unimodality | Not satisfied | ✗ |
| Low skewness | Not satisfied | ✗ |
| Near-normal kurtosis | Satisfied | ✓ |
| High separability | Yes | ✓ |

### 4. Conclusion

**Assessment**

The Hepatitis dataset presents a conflicting profile for centroid-based classification. Class separability is high and kurtosis behaviour is favourable, suggesting that the two classes are distinguishable by their centroids. However, the dataset fails three critical NC assumptions: multivariate normality is absent in both classes, unimodality is not achieved (particularly severe in class 2), and skewness exceeds acceptable thresholds. The presence of binary and categorical features in this medical dataset likely contributes to these violations, as such features inherently produce non-normal, multimodal distributions.

Therefore, observing similar performance between KNN and NC on this dataset should not be interpreted as evidence that both algorithms perform similarly in general. While high separability may enable reasonable NC performance, the fundamental structural violations of NC assumptions suggest that any comparable results are dataset-specific and should not be generalised.