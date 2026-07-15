# Dataset Analysis Lenses

1. Dataset summary
2. Statistical analysis
3. NC suitability assessment
4. Overall bias score (optional)
5. Conclusion

---

## Dataset: Lenses
> Definition 1 (NC-favourable class). In this study, a class is termed NC-favourable if it satisfies the threshold defined for the corresponding statistical property (e.g., ≥70% unimodal features, ≥60% low-skewness features, etc.). The term does not imply that NC will necessarily outperform KNN; rather, it indicates that the class exhibits characteristics consistent with the assumptions of the Nearest Centroid classifier.

### 1. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | Lenses |
| Samples | 24 |
| Features | 4 |
| Classes | 3 |
| Samples per class | 1: 4, 2: 5, 3: 15 |

### 2. Statistical Analysis

#### 2.1 Multivariate Normality (Henze-Zirkler)

| Class | HZ Statistic | p-value | Normal Distribution |
|-------|--------------|---------|---------------------|
| 1 | None | None | — N/A |
| 2 | 20.0 | 0.0 | ✗ No |
| 3 | 1.3161 | 0.0001 | ✗ No |

**Interpretation**

No class satisfies multivariate normality. Class 1 could not be evaluated due to insufficient sample size (4 samples). Classes 2 and 3 both show significant deviation from the multivariate normal distribution. The dataset does not conform to the centroid assumption used by Nearest Centroid.

#### 2.2 Unimodality (Dip Test)
> Note. Nearest Centroid (NC) assumes that each class can be reasonably represented by a single centroid. To evaluate this assumption, the percentage of unimodal features within each class is calculated using Hartigan's Dip Test (α = 0.05). A class is considered NC-favourable when 70% or more of its features are unimodal; otherwise, it is considered to exhibit multimodal structure that may reduce centroid representativeness.

| Class | Unimodal Features | Interpretation |
|-------|-------------------|----------------|
| 1 | 0.0% | Only 0% of features are unimodal — multimodal structure detected. |
| 2 | 66.7% | Only 67% of features are unimodal — multimodal structure detected. |
| 3 | 33.3% | Only 33% of features are unimodal — multimodal structure detected. |

**Interpretation**

No class meets the 70% unimodality threshold. Class 1 shows complete multimodality (0% unimodal features), class 2 falls just short (66.7%), and class 3 exhibits substantial multimodal structure (33.3%). These results indicate that a single centroid is unlikely to adequately represent any class in this dataset. The small sample sizes and categorical nature of the features likely contribute to this outcome.

#### 2.3 Feature Skewness
> Note. NC represents each class using its arithmetic mean (centroid). Highly skewed feature distributions can shift the mean away from the region containing most observations, making the centroid less representative. For each class, the percentage of features whose absolute skewness is below 0.5 is calculated. A class is considered NC-favourable when 60% or more of its features have low skewness; otherwise, asymmetric distributions are considered likely to reduce centroid reliability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| 1 | 0.0% | Only 0% of features have low skewness — asymmetric distributions detected. |
| 2 | 33.3% | Only 33% of features have low skewness — asymmetric distributions detected. |
| 3 | 66.7% | 67% of features have low skewness — data is NC-favourable. |

**Interpretation**

Only class 3 meets the NC-favourable threshold for low skewness. Classes 1 and 2 fall well below, with class 1 showing 0% near-normal features, indicating extreme asymmetry. The centroids for classes 1 and 2 are likely to be substantially distorted by skewed distributions, further undermining the NC assumption.

#### 2.4 Kurtosis
> Note. NC performs best when feature distributions are not dominated by heavy tails or extreme outliers, as such observations can distort the centroid. For each class, the percentage of features whose kurtosis falls within the predefined near-normal range is calculated. A class is considered NC-favourable when 60% or more of its features exhibit near-normal kurtosis; otherwise, heavy-tailed distributions are considered likely to reduce centroid stability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| 1 | 33.3% | Only 33% of features have near-normal kurtosis — heavy/light tails detected. |
| 2 | 33.3% | Only 33% of features have near-normal kurtosis — heavy/light tails detected. |
| 3 | 100.0% | 100% of features have near-normal kurtosis — tail behaviour is NC-favourable. |

**Interpretation**

Only class 3 satisfies the NC-favourable threshold for near-normal kurtosis, achieving a perfect 100%. Classes 1 and 2 fall substantially below the threshold at 33.3%, indicating problematic tail behaviour that may further destabilise centroid estimation for these classes.

#### 2.5 Class Separability
> Note. NC assigns a sample to the nearest class centroid. Therefore, its performance depends on the distance between class centroids relative to the spread of each class. The separability ratio is defined as the inter-class centroid distance divided by the average within-class dispersion. Larger ratios indicate better-separated classes and are therefore more favourable for NC. The following interpretation is used throughout this study:

- Ratio < 1.0: Low separability (NC-unfavourable)
- 1.0 ≤ Ratio < 1.5: Moderate separability
- Ratio ≥ 1.5: High separability (NC-favourable)

| Class Pair | Ratio | Separation |
|------------|-------|-------------|
| 1 vs 2 | 3.034 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| 1 vs 3 | 1.492 | MODERATE — Moderate separation; NC likely works but is not conclusive. |
| 2 vs 3 | 1.450 | MODERATE — Moderate separation; NC likely works but is not conclusive. |

**Average Separability Ratio**

1.992

**Interpretation**

Class separability is mixed. One pair (1 vs 2) achieves high separability, while the remaining two pairs fall into the moderate range. Although the average ratio exceeds the high separability threshold, the moderate separation for pairs involving class 3 — the largest class — is concerning for NC reliability.

### 3. Overall Dataset Assessment

| Criterion | Result | Supports NC? |
|-----------|--------|--------------|
| Multivariate normality | Not satisfied | ✗ |
| Unimodality | Not satisfied | ✗ |
| Low skewness | Mostly unsatisfied | ✗ |
| Near-normal kurtosis | Partially satisfied | ~ |
| High separability | Mixed | ~ |

### 4. Conclusion

**Assessment**

The Lenses dataset is fundamentally unsuitable for centroid-based classification. It fails nearly all NC assumptions: no class satisfies multivariate normality, unimodality is absent across all classes, and skewness is problematic for two of three classes. Class 1 is particularly problematic across all criteria, though its extremely small sample size (4) makes statistical assessments unreliable. The dataset's small overall size (24 samples), limited features (4), and categorical nature inherently violate the distributional assumptions of NC. While class separability is moderate to high and class 3 shows some favourable properties (skewness, kurtosis), these isolated positives are insufficient to overcome the widespread structural violations.

Therefore, observing similar performance between KNN and NC on this dataset should not be interpreted as evidence that both algorithms perform similarly in general. The dataset's characteristics are fundamentally misaligned with NC assumptions, and any comparable performance would be an artefact of the dataset's small size and simplicity rather than evidence of algorithmic equivalence. This dataset serves as a clear example of conditions where NC assumptions are systematically violated.