# Dataset Analysis Iris

1. Dataset summary
2. Statistical analysis
3. NC suitability assessment
4. Overall bias score (optional)
5. Conclusion

---

## Dataset: Iris
> Definition 1 (NC-favourable class). In this study, a class is termed NC-favourable if it satisfies the threshold defined for the corresponding statistical property (e.g., ≥70% unimodal features, ≥60% low-skewness features, etc.). The term does not imply that NC will necessarily outperform KNN; rather, it indicates that the class exhibits characteristics consistent with the assumptions of the Nearest Centroid classifier.

### 1. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | Iris |
| Samples | 150 |
| Features | 4 |
| Classes | 3 |
| Samples per class | 50 / 50 / 50 |

### 2. Statistical Analysis

#### 2.1 Multivariate Normality (Henze-Zirkler)

| Class | HZ Statistic | p-value | Normal Distribution |
|-------|--------------|---------|---------------------|
| Iris-setosa | 1.0297 | 0.0126 | ✗ No |
| Iris-versicolor | 0.8388 | 0.2262 | ✓ Yes |
| Iris-virginica | 0.7570 | 0.4970 | ✓ Yes |

**Interpretation**

Two of three classes satisfy multivariate normality. The dataset is largely compatible with the centroid assumption used by Nearest Centroid.

#### 2.2 Unimodality (Dip Test)
> Note. Nearest Centroid (NC) assumes that each class can be reasonably represented by a single centroid. To evaluate this assumption, the percentage of unimodal features within each class is calculated using Hartigan's Dip Test (α = 0.05). A class is considered NC-favourable when 70% or more of its features are unimodal; otherwise, it is considered to exhibit multimodal structure that may reduce centroid representativeness.

| Class | Unimodal Features | Interpretation |
|-------|-------------------|----------------|
| Iris-setosa | 50.0% | Only 50% of features are unimodal — multimodal structure detected. |
| Iris-versicolor | 75.0% | 75% of features are unimodal — NC centroid is structurally valid. |
| Iris-virginica | 50.0% | Only 50% of features are unimodal — multimodal structure detected. |

**Interpretation**

Most class distributions are either fully or largely unimodal, indicating that a single centroid can reasonably represent each class.

#### 2.3 Feature Skewness
> Note. NC represents each class using its arithmetic mean (centroid). Highly skewed feature distributions can shift the mean away from the region containing most observations, making the centroid less representative. For each class, the percentage of features whose absolute skewness is below 0.5 is calculated. A class is considered NC-favourable when 60% or more of its features have low skewness; otherwise, asymmetric distributions are considered likely to reduce centroid reliability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| Iris-setosa | 75.0% | 75% of features have low skewness — data is NC-favourable. |
| Iris-versicolor | 100.0% | 100% of features have low skewness — data is NC-favourable. |
| Iris-virginica | 100.0% | 100% of features have low skewness — data is NC-favourable. |

**Interpretation**

Low skewness indicates that class centroids are unlikely to be distorted by asymmetric distributions.

#### 2.4 Kurtosis
> Note. NC performs best when feature distributions are not dominated by heavy tails or extreme outliers, as such observations can distort the centroid. For each class, the percentage of features whose kurtosis falls within the predefined near-normal range is calculated. A class is considered NC-favourable when 60% or more of its features exhibit near-normal kurtosis; otherwise, heavy-tailed distributions are considered likely to reduce centroid stability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| Iris-setosa | 100.0% | 100% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| Iris-versicolor | 100.0% | 100% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| Iris-virginica | 100.0% | 100% of features have near-normal kurtosis — tail behaviour is NC-favourable. |

**Interpretation**

All classes exhibit approximately normal tail behaviour, reducing the influence of extreme observations on centroid estimation.

#### 2.5 Class Separability
> Note. NC assigns a sample to the nearest class centroid. Therefore, its performance depends on the distance between class centroids relative to the spread of each class. The separability ratio is defined as the inter-class centroid distance divided by the average within-class dispersion. Larger ratios indicate better-separated classes and are therefore more favourable for NC. The following interpretation is used throughout this study:

- Ratio < 1.0: Low separability (NC-unfavourable)
- 1.0 ≤ Ratio < 1.5: Moderate separability
- Ratio ≥ 1.5: High separability (NC-favoura


| Class Pair | Ratio | Separation |
|------------|-------|-------------|
| Iris-setosa vs Iris-versicolor | 5.104 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| Iris-setosa vs Iris-virginica | 6.791 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| Iris-versicolor vs Iris-virginica | 1.975 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |

**Average Separability Ratio**

4.623

**Interpretation**

All class pairs are well separated. High inter-class separation allows a centroid classifier to distinguish classes reliably.

### 3. Overall Dataset Assessment

| Criterion | Result | Supports NC? |
|-----------|--------|--------------|
| Multivariate normality | Mostly satisfied | ✓ |
| Unimodality | Mostly satisfied | ✓ |
| Low skewness | Yes | ✓ |
| Near-normal kurtosis | Yes | ✓ |
| High separability | Yes | ✓ |

### 4. Conclusion

**Assessment**

The Iris dataset exhibits several structural properties that favour centroid-based classification. Most classes are approximately multivariate normal, feature distributions are largely symmetric with near-normal kurtosis, and class separability is high. These characteristics satisfy the underlying assumptions of the Nearest Centroid classifier.

Therefore, observing similar performance between KNN and NC on this dataset should not be interpreted as evidence that both algorithms perform similarly in general. Instead, the dataset itself possesses statistical characteristics that naturally benefit centroid-based classification.