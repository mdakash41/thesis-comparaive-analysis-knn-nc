# Dataset Analysis Lung Cancer

1. Dataset summary
2. Statistical analysis
3. NC suitability assessment
4. Overall bias score (optional)
5. Conclusion

---

## Dataset: Lung Cancer
> Definition 1 (NC-favourable class). In this study, a class is termed NC-favourable if it satisfies the threshold defined for the corresponding statistical property (e.g., ≥70% unimodal features, ≥60% low-skewness features, etc.). The term does not imply that NC will necessarily outperform KNN; rather, it indicates that the class exhibits characteristics consistent with the assumptions of the Nearest Centroid classifier.

### 1. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | Lung Cancer |
| Samples | 32 |
| Features | 56 |
| Classes | 3 |
| Samples per class | 1: 9, 2: 13, 3: 10 |

### 2. Statistical Analysis

#### 2.1 Multivariate Normality (Henze-Zirkler)

| Class | HZ Statistic | p-value | Normal Distribution |
|-------|--------------|---------|---------------------|
| 1 | 32.0 | 0.0 | ✗ No |
| 2 | 40.0 | 0.0 | ✗ No |
| 3 | 36.0 | 0.0 | ✗ No |

**Interpretation**

No class satisfies multivariate normality. All three classes show highly significant deviations, indicating that the dataset does not conform to the centroid assumption used by Nearest Centroid. The high-dimensional nature of this dataset (56 features) combined with very small sample sizes (9–13 per class) likely contributes to this outcome.

#### 2.2 Unimodality (Dip Test)
> Note. Nearest Centroid (NC) assumes that each class can be reasonably represented by a single centroid. To evaluate this assumption, the percentage of unimodal features within each class is calculated using Hartigan's Dip Test (α = 0.05). A class is considered NC-favourable when 70% or more of its features are unimodal; otherwise, it is considered to exhibit multimodal structure that may reduce centroid representativeness.

| Class | Unimodal Features | Interpretation |
|-------|-------------------|----------------|
| 1 | 67.9% | Only 68% of features are unimodal — multimodal structure detected. |
| 2 | 50.0% | Only 50% of features are unimodal — multimodal structure detected. |
| 3 | 58.9% | Only 59% of features are unimodal — multimodal structure detected. |

**Interpretation**

No class meets the 70% unimodality threshold. Class 1 falls just short (67.9%), while classes 2 and 3 show more substantial multimodal structure (50.0% and 58.9% respectively). This indicates that a single centroid cannot adequately represent any class, with the problem most severe for class 2 where half of all features are multimodal.

#### 2.3 Feature Skewness
> Note. NC represents each class using its arithmetic mean (centroid). Highly skewed feature distributions can shift the mean away from the region containing most observations, making the centroid less representative. For each class, the percentage of features whose absolute skewness is below 0.5 is calculated. A class is considered NC-favourable when 60% or more of its features have low skewness; otherwise, asymmetric distributions are considered likely to reduce centroid reliability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| 1 | 37.5% | Only 38% of features have low skewness — asymmetric distributions detected. |
| 2 | 51.8% | Only 52% of features have low skewness — asymmetric distributions detected. |
| 3 | 62.5% | 62% of features have low skewness — data is NC-favourable. |

**Interpretation**

Only class 3 meets the NC-favourable threshold for low skewness. Classes 1 and 2 fall substantially below, with class 1 showing particularly severe asymmetry (only 37.5% near-normal features). Centroids for classes 1 and 2 are likely to be distorted by skewed feature distributions, while class 3's centroid appears more reliable in this respect.

#### 2.4 Kurtosis
> Note. NC performs best when feature distributions are not dominated by heavy tails or extreme outliers, as such observations can distort the centroid. For each class, the percentage of features whose kurtosis falls within the predefined near-normal range is calculated. A class is considered NC-favourable when 60% or more of its features exhibit near-normal kurtosis; otherwise, heavy-tailed distributions are considered likely to reduce centroid stability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| 1 | 51.8% | Only 52% of features have near-normal kurtosis — heavy/light tails detected. |
| 2 | 64.3% | 64% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| 3 | 82.1% | 82% of features have near-normal kurtosis — tail behaviour is NC-favourable. |

**Interpretation**

Classes 2 and 3 satisfy the NC-favourable threshold for near-normal kurtosis, with class 3 performing particularly well (82.1%). Class 1 falls below the threshold at 51.8%, indicating that heavy or light tails may influence centroid estimation for this class.

#### 2.5 Class Separability
> Note. NC assigns a sample to the nearest class centroid. Therefore, its performance depends on the distance between class centroids relative to the spread of each class. The separability ratio is defined as the inter-class centroid distance divided by the average within-class dispersion. Larger ratios indicate better-separated classes and are therefore more favourable for NC. The following interpretation is used throughout this study:

- Ratio < 1.0: Low separability (NC-unfavourable)
- 1.0 ≤ Ratio < 1.5: Moderate separability
- Ratio ≥ 1.5: High separability (NC-favourable)

| Class Pair | Ratio | Separation |
|------------|-------|-------------|
| 1 vs 2 | 2.620 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 1 vs 3 | 4.060 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 2 vs 3 | 2.554 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |

**Average Separability Ratio**

3.078

**Interpretation**

All class pairs exhibit high separability, with ratios well above the NC-favourable threshold. Class pair 1 vs 3 shows particularly strong separation (4.060). The consistently high separability suggests that class centroids occupy distinct regions in the feature space, which is favourable for centroid-based discrimination.

### 3. Overall Dataset Assessment

| Criterion | Result | Supports NC? |
|-----------|--------|--------------|
| Multivariate normality | Not satisfied | ✗ |
| Unimodality | Not satisfied | ✗ |
| Low skewness | Mostly unsatisfied | ✗ |
| Near-normal kurtosis | Partially satisfied | ~ |
| High separability | Strongly satisfied | ✓ |

### 4. Conclusion

**Assessment**

The Lung Cancer dataset presents a challenging and largely unfavourable profile for centroid-based classification. Despite strong class separability across all pairs, the dataset fails three critical NC assumptions: multivariate normality is absent, unimodality is not achieved by any class, and skewness is problematic for two of three classes. The severe mismatch between the number of features (56) and samples (32) creates a high-dimensional, low-sample-size scenario that inherently undermines distributional assumptions. Class 3 emerges as the most NC-favourable, meeting thresholds for both skewness and kurtosis, while class 1 is the most problematic across nearly all criteria.

Therefore, observing similar performance between KNN and NC on this dataset should not be interpreted as evidence that both algorithms perform similarly in general. The dataset's structural properties substantially violate NC assumptions, and the high dimensionality with small sample sizes creates conditions where centroid estimation is inherently unstable. Any comparable performance would reflect dataset-specific artefacts rather than algorithmic equivalence.