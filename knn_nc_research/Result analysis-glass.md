# Dataset Analysis Glass

1. Dataset summary
2. Statistical analysis
3. NC suitability assessment
4. Overall bias score (optional)
5. Conclusion

---

## Dataset: Glass
> Definition 1 (NC-favourable class). In this study, a class is termed NC-favourable if it satisfies the threshold defined for the corresponding statistical property (e.g., ≥70% unimodal features, ≥60% low-skewness features, etc.). The term does not imply that NC will necessarily outperform KNN; rather, it indicates that the class exhibits characteristics consistent with the assumptions of the Nearest Centroid classifier.

### 1. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | Glass |
| Samples | 214 |
| Features | 9 |
| Classes | 6 |
| Samples per class | 1: 70, 2: 76, 3: 17, 5: 13, 6: 9, 7: 29 |

### 2. Statistical Analysis

#### 2.1 Multivariate Normality (Henze-Zirkler)

| Class | HZ Statistic | p-value | Normal Distribution |
|-------|--------------|---------|---------------------|
| 1 | 1.7482 | 0.0 | ✗ No |
| 2 | 2.4611 | 0.0 | ✗ No |
| 3 | 0.9579 | 0.1324 | ✓ Yes |
| 5 | 0.9639 | 0.0599 | ✓ Yes |
| 6 | 36.0 | 0.0 | ✗ No |
| 7 | 1.3413 | 0.0 | ✗ No |

**Interpretation**

Two of six classes (3 and 5) satisfy multivariate normality. The remaining four classes deviate significantly, indicating that the dataset only partially conforms to the centroid assumption used by Nearest Centroid.

#### 2.2 Unimodality (Dip Test)
> Note. Nearest Centroid (NC) assumes that each class can be reasonably represented by a single centroid. To evaluate this assumption, the percentage of unimodal features within each class is calculated using Hartigan's Dip Test (α = 0.05). A class is considered NC-favourable when 70% or more of its features are unimodal; otherwise, it is considered to exhibit multimodal structure that may reduce centroid representativeness.

| Class | Unimodal Features | Interpretation |
|-------|-------------------|----------------|
| 1 | 88.9% | 89% of features are unimodal — NC centroid is structurally valid. |
| 2 | 77.8% | 78% of features are unimodal — NC centroid is structurally valid. |
| 3 | 100.0% | 100% of features are unimodal — NC centroid is structurally valid. |
| 5 | 88.9% | 89% of features are unimodal — NC centroid is structurally valid. |
| 6 | 88.9% | 89% of features are unimodal — NC centroid is structurally valid. |
| 7 | 88.9% | 89% of features are unimodal — NC centroid is structurally valid. |

**Interpretation**

All six classes satisfy the unimodality threshold, with most classes exceeding 88% unimodal features. This indicates that a single centroid can reasonably represent each class, strongly supporting the NC assumption at the class level.

#### 2.3 Feature Skewness
> Note. NC represents each class using its arithmetic mean (centroid). Highly skewed feature distributions can shift the mean away from the region containing most observations, making the centroid less representative. For each class, the percentage of features whose absolute skewness is below 0.5 is calculated. A class is considered NC-favourable when 60% or more of its features have low skewness; otherwise, asymmetric distributions are considered likely to reduce centroid reliability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| 1 | 66.7% | 67% of features have low skewness — data is NC-favourable. |
| 2 | 33.3% | Only 33% of features have low skewness — asymmetric distributions detected. |
| 3 | 66.7% | 67% of features have low skewness — data is NC-favourable. |
| 5 | 44.4% | Only 44% of features have low skewness — asymmetric distributions detected. |
| 6 | 33.3% | Only 33% of features have low skewness — asymmetric distributions detected. |
| 7 | 22.2% | Only 22% of features have low skewness — asymmetric distributions detected. |

**Interpretation**

Only two classes (1 and 3) meet the NC-favourable threshold for low skewness. The remaining four classes exhibit substantial asymmetry, with class 7 showing particularly high skewness (only 22% near-normal features). This suggests that centroids may be distorted by skewed feature distributions for the majority of classes.

#### 2.4 Kurtosis
> Note. NC performs best when feature distributions are not dominated by heavy tails or extreme outliers, as such observations can distort the centroid. For each class, the percentage of features whose kurtosis falls within the predefined near-normal range is calculated. A class is considered NC-favourable when 60% or more of its features exhibit near-normal kurtosis; otherwise, heavy-tailed distributions are considered likely to reduce centroid stability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| 1 | 77.8% | 78% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| 2 | 44.4% | Only 44% of features have near-normal kurtosis — heavy/light tails detected. |
| 3 | 77.8% | 78% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| 5 | 77.8% | 78% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| 6 | 55.6% | Only 56% of features have near-normal kurtosis — heavy/light tails detected. |
| 7 | 55.6% | Only 56% of features have near-normal kurtosis — heavy/light tails detected. |

**Interpretation**

Three classes (1, 3, 5) meet the NC-favourable threshold for near-normal kurtosis. The remaining three classes (2, 6, 7) fall below the threshold, indicating that heavy or light tails may influence centroid estimation. Class 2 is the most affected, with only 44% of features exhibiting near-normal kurtosis.

#### 2.5 Class Separability
> Note. NC assigns a sample to the nearest class centroid. Therefore, its performance depends on the distance between class centroids relative to the spread of each class. The separability ratio is defined as the inter-class centroid distance divided by the average within-class dispersion. Larger ratios indicate better-separated classes and are therefore more favourable for NC. The following interpretation is used throughout this study:

- Ratio < 1.0: Low separability (NC-unfavourable)
- 1.0 ≤ Ratio < 1.5: Moderate separability
- Ratio ≥ 1.5: High separability (NC-favourable)

| Class Pair | Ratio | Separation |
|------------|-------|-------------|
| 1 vs 2 | 0.758 | LOW — Poor separation; NC centroid assumption is under stress. |
| 1 vs 3 | 0.549 | LOW — Poor separation; NC centroid assumption is under stress. |
| 1 vs 5 | 2.684 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 1 vs 6 | 3.232 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 1 vs 7 | 3.974 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 2 vs 3 | 0.876 | LOW — Poor separation; NC centroid assumption is under stress. |
| 2 vs 5 | 1.716 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 2 vs 6 | 2.033 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 2 vs 7 | 2.554 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 3 vs 5 | 2.766 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 3 vs 6 | 3.286 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 3 vs 7 | 4.051 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 5 vs 6 | 1.748 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 5 vs 7 | 1.742 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 6 vs 7 | 1.488 | MODERATE — Moderate separation; NC likely works but is not conclusive. |

**Average Separability Ratio**

2.23

**Interpretation**

While the average separability ratio is 2.23 (high), three class pairs fall into the low separability category (1 vs 2, 1 vs 3, 2 vs 3) and one pair shows moderate separation (6 vs 7). These poorly separated pairs may present classification challenges for NC, as overlapping class distributions reduce centroid discriminability.

### 3. Overall Dataset Assessment

| Criterion | Result | Supports NC? |
|-----------|--------|--------------|
| Multivariate normality | Partially satisfied | ~ |
| Unimodality | Strongly satisfied | ✓ |
| Low skewness | Mostly unsatisfied | ✗ |
| Near-normal kurtosis | Partially satisfied | ~ |
| High separability | Mixed | ~ |

### 4. Conclusion

**Assessment**

The Glass dataset presents a complex and mixed profile for centroid-based classification. Unimodality is strongly satisfied across all classes, providing good support for the single-centroid assumption. However, other criteria are substantially violated: most classes fail multivariate normality, four of six classes exhibit problematic skewness, and three class pairs show low separability. The combination of asymmetric feature distributions and overlapping classes (particularly among classes 1, 2, and 3) suggests that NC may struggle in regions where these classes intersect.

Therefore, observing similar performance between KNN and NC on this dataset should not be interpreted as evidence that both algorithms perform similarly in general. The dataset exhibits structural characteristics that violate key NC assumptions, and any comparable performance would likely reflect dataset-specific properties rather than algorithmic equivalence.