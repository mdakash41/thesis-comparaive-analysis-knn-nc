# Dataset Analysis Ecoli

1. Dataset summary
2. Statistical analysis
3. NC suitability assessment
4. Overall bias score (optional)
5. Conclusion

---

## Dataset: Ecoli
> Definition 1 (NC-favourable class). In this study, a class is termed NC-favourable if it satisfies the threshold defined for the corresponding statistical property (e.g., ≥70% unimodal features, ≥60% low-skewness features, etc.). The term does not imply that NC will necessarily outperform KNN; rather, it indicates that the class exhibits characteristics consistent with the assumptions of the Nearest Centroid classifier.

### 1. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | Ecoli |
| Samples | 336 |
| Features | 7 |
| Classes | 8 |
| Samples per class | cp: 143, im: 77, imL: 2, imS: 2, imU: 35, om: 20, omL: 5, pp: 52 |

### 2. Statistical Analysis

#### 2.1 Multivariate Normality (Henze-Zirkler)

| Class | HZ Statistic | p-value | Normal Distribution |
|-------|--------------|---------|---------------------|
| cp | 572.0 | 0.0 | ✗ No |
| im | 308.0 | 0.0 | ✗ No |
| imL | None | None | — N/A |
| imS | None | None | — N/A |
| imU | 140.0 | 0.0 | ✗ No |
| om | 80.0 | 0.0 | ✗ No |
| omL | 20.0 | 0.0 | ✗ No |
| pp | 208.0 | 0.0 | ✗ No |

**Interpretation**

No class satisfies multivariate normality. The dataset does not conform to the centroid assumption used by Nearest Centroid. Classes imL and imS could not be evaluated due to insufficient sample size.

#### 2.2 Unimodality (Dip Test)
> Note. Nearest Centroid (NC) assumes that each class can be reasonably represented by a single centroid. To evaluate this assumption, the percentage of unimodal features within each class is calculated using Hartigan's Dip Test (α = 0.05). A class is considered NC-favourable when 70% or more of its features are unimodal; otherwise, it is considered to exhibit multimodal structure that may reduce centroid representativeness.

| Class | Unimodal Features | Interpretation |
|-------|-------------------|----------------|
| cp | 100.0% | 100% of features are unimodal — NC centroid is structurally valid. |
| im | 100.0% | 100% of features are unimodal — NC centroid is structurally valid. |
| imL | 0.0% | Only 0% of features are unimodal — multimodal structure detected. |
| imS | 0.0% | Only 0% of features are unimodal — multimodal structure detected. |
| imU | 100.0% | 100% of features are unimodal — NC centroid is structurally valid. |
| om | 100.0% | 100% of features are unimodal — NC centroid is structurally valid. |
| omL | 100.0% | 100% of features are unimodal — NC centroid is structurally valid. |
| pp | 100.0% | 100% of features are unimodal — NC centroid is structurally valid. |

**Interpretation**

Six of eight classes are fully unimodal, indicating that a single centroid can reasonably represent most classes. Classes imL and imS exhibit complete multimodal structure, likely due to extremely small sample sizes (2 samples each).

#### 2.3 Feature Skewness
> Note. NC represents each class using its arithmetic mean (centroid). Highly skewed feature distributions can shift the mean away from the region containing most observations, making the centroid less representative. For each class, the percentage of features whose absolute skewness is below 0.5 is calculated. A class is considered NC-favourable when 60% or more of its features have low skewness; otherwise, asymmetric distributions are considered likely to reduce centroid reliability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| cp | 71.4% | 71% of features have low skewness — data is NC-favourable. |
| im | 42.9% | Only 43% of features have low skewness — asymmetric distributions detected. |
| imL | 0.0% | Only 0% of features have low skewness — asymmetric distributions detected. |
| imS | 0.0% | Only 0% of features have low skewness — asymmetric distributions detected. |
| imU | 42.9% | Only 43% of features have low skewness — asymmetric distributions detected. |
| om | 71.4% | 71% of features have low skewness — data is NC-favourable. |
| omL | 71.4% | 71% of features have low skewness — data is NC-favourable. |
| pp | 28.6% | Only 29% of features have low skewness — asymmetric distributions detected. |

**Interpretation**

Only three of eight classes (cp, om, omL) meet the NC-favourable threshold for low skewness. The remaining classes exhibit substantial asymmetry, indicating that centroids may be distorted by skewed feature distributions. Caution is warranted for classes imL, imS, and pp due to very low sample sizes.

#### 2.4 Kurtosis
> Note. NC performs best when feature distributions are not dominated by heavy tails or extreme outliers, as such observations can distort the centroid. For each class, the percentage of features whose kurtosis falls within the predefined near-normal range is calculated. A class is considered NC-favourable when 60% or more of its features exhibit near-normal kurtosis; otherwise, heavy-tailed distributions are considered likely to reduce centroid stability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| cp | 71.4% | 71% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| im | 42.9% | Only 43% of features have near-normal kurtosis — heavy/light tails detected. |
| imL | 0.0% | Only 0% of features have near-normal kurtosis — heavy/light tails detected. |
| imS | 0.0% | Only 0% of features have near-normal kurtosis — heavy/light tails detected. |
| imU | 42.9% | Only 43% of features have near-normal kurtosis — heavy/light tails detected. |
| om | 71.4% | 71% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| omL | 71.4% | 71% of features have near-normal kurtosis — tail behaviour is NC-favourable. |
| pp | 28.6% | Only 29% of features have near-normal kurtosis — heavy/light tails detected. |

**Interpretation**

Three classes (cp, om, omL) exhibit predominantly normal tail behaviour. The remaining five classes show non-normal kurtosis, suggesting that extreme observations may influence centroid estimation. Classes imL and imS show 0% near-normal kurtosis, though results may be unreliable due to extremely small sample sizes.

#### 2.5 Class Separability
> Note. NC assigns a sample to the nearest class centroid. Therefore, its performance depends on the distance between class centroids relative to the spread of each class. The separability ratio is defined as the inter-class centroid distance divided by the average within-class dispersion. Larger ratios indicate better-separated classes and are therefore more favourable for NC. The following interpretation is used throughout this study:

- Ratio < 1.0: Low separability (NC-unfavourable)
- 1.0 ≤ Ratio < 1.5: Moderate separability
- Ratio ≥ 1.5: High separability (NC-favourable)

| Class Pair | Ratio | Separation |
|------------|-------|-------------|
| cp vs im | 3.327 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| cp vs imL | 3.200 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| cp vs imS | 3.259 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| cp vs imU | 4.374 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| cp vs om | 3.584 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| cp vs omL | 5.727 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| cp vs pp | 3.007 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| im vs imL | 2.378 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| im vs imS | 1.647 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| im vs imU | 1.350 | MODERATE — Moderate separation; NC likely works but is not conclusive. |
| im vs om | 3.385 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| im vs omL | 5.015 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| im vs pp | 3.024 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imL vs imS | 2.174 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imL vs imU | 2.376 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imL vs om | 2.921 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imL vs omL | 1.994 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imL vs pp | 2.741 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imS vs imU | 1.198 | MODERATE — Moderate separation; NC likely works but is not conclusive. |
| imS vs om | 2.448 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imS vs omL | 4.278 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imS vs pp | 2.038 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imU vs om | 3.712 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imU vs omL | 5.582 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| imU vs pp | 3.429 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| om vs omL | 4.383 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| om vs pp | 1.997 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| omL vs pp | 4.681 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |

**Average Separability Ratio**

3.187

**Interpretation**

The vast majority of class pairs (26 of 28) exhibit high separability. Two pairs (im vs imU and imS vs imU) show moderate separation. The high average separability ratio of 3.187 suggests that class centroids are generally well-distinguished.

### 3. Overall Dataset Assessment

| Criterion | Result | Supports NC? |
|-----------|--------|--------------|
| Multivariate normality | Not satisfied | ✗ |
| Unimodality | Mostly satisfied | ✓ |
| Low skewness | Partially satisfied | ~ |
| Near-normal kurtosis | Partially satisfied | ~ |
| High separability | Yes | ✓ |

### 4. Conclusion

**Assessment**

The Ecoli dataset presents a mixed profile for centroid-based classification. While class separability is high and most classes exhibit unimodal feature distributions, the dataset fails multivariate normality tests entirely, and only three of eight classes meet the skewness and kurtosis thresholds for NC-favourability. Several classes (imL, imS) have extremely small sample sizes, making their statistical assessments unreliable. 

Therefore, observing similar performance between KNN and NC on this dataset should not be interpreted as evidence that both algorithms perform similarly in general. The dataset's structural properties only partially align with Nearest Centroid assumptions, and performance differences may be attributable to these violations rather than algorithmic equivalence.