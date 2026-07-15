# Dataset Analysis Soybean

1. Dataset summary
2. Statistical analysis
3. NC suitability assessment
4. Overall bias score (optional)
5. Conclusion

---

## Dataset: Soybean
> Definition 1 (NC-favourable class). In this study, a class is termed NC-favourable if it satisfies the threshold defined for the corresponding statistical property (e.g., ≥70% unimodal features, ≥60% low-skewness features, etc.). The term does not imply that NC will necessarily outperform KNN; rather, it indicates that the class exhibits characteristics consistent with the assumptions of the Nearest Centroid classifier.

### 1. Dataset Overview

| Property | Value |
|----------|-------|
| Dataset | Soybean |
| Samples | 47 |
| Features | 35 |
| Classes | 4 |
| Samples per class | D1: 10, D2: 10, D3: 10, D4: 17 |

### 2. Statistical Analysis

#### 2.1 Multivariate Normality (Henze-Zirkler)

| Class | HZ Statistic | p-value | Normal Distribution |
|-------|--------------|---------|---------------------|
| D1 | 40.0 | 0.0 | ✗ No |
| D2 | 40.0 | 0.0 | ✗ No |
| D3 | 40.0 | 0.0 | ✗ No |
| D4 | 68.0 | 0.0 | ✗ No |

**Interpretation**

No class satisfies multivariate normality. All four classes show highly significant deviations from the multivariate normal distribution. The dataset does not conform to the centroid assumption used by Nearest Centroid.

#### 2.2 Unimodality (Dip Test)
> Note. Nearest Centroid (NC) assumes that each class can be reasonably represented by a single centroid. To evaluate this assumption, the percentage of unimodal features within each class is calculated using Hartigan's Dip Test (α = 0.05). A class is considered NC-favourable when 70% or more of its features are unimodal; otherwise, it is considered to exhibit multimodal structure that may reduce centroid representativeness.

| Class | Unimodal Features | Interpretation |
|-------|-------------------|----------------|
| D1 | 77.1% | 77% of features are unimodal — NC centroid is structurally valid. |
| D2 | 77.1% | 77% of features are unimodal — NC centroid is structurally valid. |
| D3 | 82.9% | 83% of features are unimodal — NC centroid is structurally valid. |
| D4 | 71.4% | 71% of features are unimodal — NC centroid is structurally valid. |

**Interpretation**

All four classes satisfy the unimodality threshold, with percentages ranging from 71.4% to 82.9%. This indicates that a single centroid can reasonably represent each class, providing strong support for the NC assumption at the class structure level. Class D3 shows the strongest unimodality, while D4 meets the threshold more narrowly.

#### 2.3 Feature Skewness
> Note. NC represents each class using its arithmetic mean (centroid). Highly skewed feature distributions can shift the mean away from the region containing most observations, making the centroid less representative. For each class, the percentage of features whose absolute skewness is below 0.5 is calculated. A class is considered NC-favourable when 60% or more of its features have low skewness; otherwise, asymmetric distributions are considered likely to reduce centroid reliability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| D1 | 22.9% | Only 23% of features have low skewness — asymmetric distributions detected. |
| D2 | 22.9% | Only 23% of features have low skewness — asymmetric distributions detected. |
| D3 | 17.1% | Only 17% of features have low skewness — asymmetric distributions detected. |
| D4 | 25.7% | Only 26% of features have low skewness — asymmetric distributions detected. |

**Interpretation**

No class comes close to meeting the NC-favourable threshold for low skewness. All classes show extremely low percentages of near-normal features (17.1%–25.7%), indicating pervasive asymmetry throughout the dataset. Centroids are highly likely to be distorted by skewed feature distributions across all classes, representing a severe violation of the NC assumption.

#### 2.4 Kurtosis
> Note. NC performs best when feature distributions are not dominated by heavy tails or extreme outliers, as such observations can distort the centroid. For each class, the percentage of features whose kurtosis falls within the predefined near-normal range is calculated. A class is considered NC-favourable when 60% or more of its features exhibit near-normal kurtosis; otherwise, heavy-tailed distributions are considered likely to reduce centroid stability.

| Class | Near-Normal Features | Assessment |
|-------|---------------------|------------|
| D1 | 20.0% | Only 20% of features have near-normal kurtosis — heavy/light tails detected. |
| D2 | 17.1% | Only 17% of features have near-normal kurtosis — heavy/light tails detected. |
| D3 | 17.1% | Only 17% of features have near-normal kurtosis — heavy/light tails detected. |
| D4 | 28.6% | Only 29% of features have near-normal kurtosis — heavy/light tails detected. |

**Interpretation**

No class satisfies the NC-favourable threshold for near-normal kurtosis. The percentages are strikingly low across all classes (17.1%–28.6%), indicating that heavy or light tails are pervasive. Extreme observations are likely to exert substantial influence on centroid estimation, further undermining the NC assumption.

#### 2.5 Class Separability
> Note. NC assigns a sample to the nearest class centroid. Therefore, its performance depends on the distance between class centroids relative to the spread of each class. The separability ratio is defined as the inter-class centroid distance divided by the average within-class dispersion. Larger ratios indicate better-separated classes and are therefore more favourable for NC. The following interpretation is used throughout this study:

- Ratio < 1.0: Low separability (NC-unfavourable)
- 1.0 ≤ Ratio < 1.5: Moderate separability
- Ratio ≥ 1.5: High separability (NC-favourable)

| Class Pair | Ratio | Separation |
|------------|-------|-------------|
| D1 vs D2 | 16.977 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| D1 vs D3 | 14.709 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| D1 vs D4 | 14.104 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| D2 vs D3 | 18.280 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| D2 vs D4 | 16.368 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |
| D3 vs D4 | 5.269 | HIGH — Classes well-separated; NC centroid is a reliable discriminator. |

**Average Separability Ratio**

14.284

**Interpretation**

Class separability is exceptionally high across all pairs, with ratios ranging from 5.269 to 18.280 — far exceeding the high separability threshold of 1.5. The average ratio of 14.284 is the highest observed among all datasets analysed. This indicates that class centroids are extremely well-separated in the feature space, providing a strong structural advantage for centroid-based discrimination despite the unfavourable within-class properties.

### 3. Overall Dataset Assessment

| Criterion | Result | Supports NC? |
|-----------|--------|--------------|
| Multivariate normality | Not satisfied | ✗ |
| Unimodality | Satisfied | ✓ |
| Low skewness | Not satisfied | ✗ |
| Near-normal kurtosis | Not satisfied | ✗ |
| High separability | Exceptionally satisfied | ✓ |

### 4. Conclusion

**Assessment**

The Soybean dataset presents a starkly divided profile for centroid-based classification. On one hand, class separability is extraordinarily high (average ratio 14.284), and unimodality is satisfied across all classes. On the other hand, the dataset fails multivariate normality, and both skewness and kurtosis are severely violated — with no class achieving even 30% near-normal features on either measure. This paradox suggests that while class centroids occupy very distinct regions of the feature space (favouring discriminability), those centroids themselves may be poor representatives of their respective class distributions due to pervasive asymmetry and non-normal tail behaviour. The high-dimensional, low-sample-size nature of the dataset (35 features, 47 samples) likely contributes to these extreme statistical properties.

Therefore, observing similar performance between KNN and NC on this dataset should not be interpreted as evidence that both algorithms perform similarly in general. Despite exceptional separability, the severe violations of skewness and kurtosis assumptions indicate that centroid representativeness is fundamentally compromised. This dataset illustrates that high separability alone does not guarantee NC suitability when within-class distributions are poorly behaved.